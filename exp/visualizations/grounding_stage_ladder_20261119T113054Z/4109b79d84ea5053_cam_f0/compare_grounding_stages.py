"""Same image, same prompts, along the whole Steer-DrivoR weight chain.

Question: the official SteerViT separates "the white car" / "the red car"; the stage-2
model does not. Where is the ability lost? Rows (each one = one set of weights):

  official  official SteerViT code + weights (SteerViT.__init__ / load_state_dict as in
            SteerViT.from_pretrained). The whole camera image is resized to 336x336 (same
            field of view as DrivoR). The demo's centre crop is NOT used: it keeps only the
            middle ~50% of the width and would cut away e.g. the red car on the left.
  pure      our encoder at 1148x672: DINOv2 + official SteerViT modules, LoRA B = 0
            (checked), no checkpoint D.          -> differs from `official` by resolution / registers
  init      checkpoint D + official SteerViT modules (= step 0 of stage 1).
                                                  -> differs from `pure` only by D (LoRA, registers)
  stage1    D + steer modules of the stage-1 checkpoint used by the stage-2 run.
  stage2    the complete stage-2 checkpoint.

pure / init / stage1 use gate scale s = 1 (as stage 1) and the CA mode stage 1 was trained
with (steer_meta, missing = true); stage2 uses its saved s and the mode of its own config.

Readouts per prompt (+ a "(no text)" column):
  seg  lin_seg_head + spatial softmax (SteerViT get_heatmap_logits; our grounding output)
  cls  last-block CLS->patch attention, head mean, official sink suppression, min-max
       (SteerViT TCAttentionExtract.get_attention_heatmaps, i.e. the demo heatmap)
Optional --box NAME=x0,y0,x1,y1 (normalised to the camera image): mass of each map in each box.

Our rows run the stage-1 path model.ground_rows (one camera row + that camera's 16 registers);
the image backbone processes every camera row independently, so this is the same per-camera
computation as the four-camera forward. FP32 everywhere; inference only.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'nuplan-devkit')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/drivor_token_viz_mpl')

import numpy as np
from PIL import Image

import visualize_steer_tokens as viz
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DISPLAY_SIZE = (1148, 672)
ROW_ORDER = ('official', 'pure', 'init', 'stage1', 'stage2')
ROW_TITLES = {
    'official': 'official SteerViT (336x336, full view)',
    'pure': 'pure: DINOv2 + SteerViT in our encoder',
    'init': 'init: D + SteerViT (stage-1 step 0)',
    'stage1': 'stage1: stage-1 checkpoint',
    'stage2': 'stage2: stage-2 checkpoint',
}


# ----------------------------------------------------------------------------- torch-free helpers
def parse_box(text):
    if '=' not in text:
        raise argparse.ArgumentTypeError(f'--box needs NAME=x0,y0,x1,y1, got {text!r}')
    name, coords = text.rsplit('=', 1)
    values = [float(v) for v in coords.split(',')]
    if len(values) != 4 or not name.strip():
        raise argparse.ArgumentTypeError(f'--box needs NAME=x0,y0,x1,y1, got {text!r}')
    x0, y0, x1, y1 = values
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise argparse.ArgumentTypeError(f'box coordinates must be normalised and ordered: {text!r}')
    return name.strip(), (x0, y0, x1, y1)


def patch_centres(grid):
    """Normalised (x, y) of every patch centre, row-major like the token order."""
    gh, gw = grid
    ys, xs = np.meshgrid((np.arange(gh) + .5) / gh, (np.arange(gw) + .5) / gw, indexing='ij')
    return np.stack([xs.ravel(), ys.ravel()], 1)


def box_masks(boxes, grid):
    c = patch_centres(grid)
    return {name: (c[:, 0] >= x0) & (c[:, 0] <= x1) & (c[:, 1] >= y0) & (c[:, 1] <= y1)
            for name, (x0, y0, x1, y1) in boxes.items()}


def total_variation(p, q):
    p, q = p / p.sum(), q / q.sum()
    return float(np.abs(p - q).sum() / 2)


def row_metrics(row, labels, boxes):
    grid = tuple(row['grid'])
    centres = patch_centres(grid)
    masks = box_masks(boxes, grid)
    out = dict(grid=list(grid), box_patches={n: int(m.sum()) for n, m in masks.items()}, columns=[], pairs=[])
    for j, label in enumerate(labels):
        seg, cls = row['seg'][j], row['cls'][j]
        cls_share = cls / max(float(cls.sum()), 1e-12)
        out['columns'].append(dict(
            text=label,
            seg_peak_xy=[round(float(v), 4) for v in centres[int(seg.argmax())]],
            cls_peak_xy=[round(float(v), 4) for v in centres[int(cls.argmax())]],
            seg_box_mass={n: float(seg[m].sum()) for n, m in masks.items()},
            cls_box_share={n: float(cls_share[m].sum()) for n, m in masks.items()}))
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            out['pairs'].append(dict(a=labels[i], b=labels[j],
                                     seg_tv=total_variation(row['seg'][i], row['seg'][j]),
                                     cls_tv=total_variation(row['cls'][i], row['cls'][j])))
    return out


def render(rows, labels, boxes, image_path, readout, path):
    image = np.asarray(Image.open(image_path).convert('RGB').resize(DISPLAY_SIZE))
    names = [n for n in ROW_ORDER if n in rows]
    fig, axes = plt.subplots(len(names), len(labels), figsize=(4.3 * len(labels), 2.9 * len(names)), squeeze=False)
    for r, name in enumerate(names):
        row = rows[name]
        maps = row[readout]
        vmax = float(np.percentile(maps, 99.5)) if readout == 'seg' else 1.0
        masks = box_masks(boxes, tuple(row['grid']))
        for j, label in enumerate(labels):
            values = maps[j]
            title = f'{name}: "{label}"'
            if boxes:
                denom = 1.0 if readout == 'seg' else max(float(values.sum()), 1e-12)
                title += '\n' + ' | '.join(f'{n} {values[m].sum() / denom:.2f}' for n, m in masks.items())
            ax = axes[r, j]
            viz.overlay(ax, image, values.reshape(row['grid']), vmax, title)
            for n, (x0, y0, x1, y1) in boxes.items():
                W, H = DISPLAY_SIZE
                ax.add_patch(Rectangle((x0 * W, y0 * H), (x1 - x0) * W, (y1 - y0) * H,
                                       fill=False, edgecolor='cyan', linewidth=1.0))
        axes[r, 0].text(-0.03, .5, ROW_TITLES[name], transform=axes[r, 0].transAxes, rotation=90,
                        ha='right', va='center', fontsize=9)
    what = ('seg = lin_seg_head spatial softmax; colour scale shared per row (99.5th pct); '
            'numbers = probability mass in each box' if readout == 'seg' else
            'cls = last-block CLS->patch attention, head mean, official sink suppression, min-max; '
            'numbers = share of the map in each box')
    fig.text(.5, .005, what, ha='center', fontsize=10)
    fig.tight_layout(rect=(0.02, .02, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def summary_markdown(run, metrics, labels, boxes):
    lines = [f"# 颜色/方位提示：沿权重链对比", '',
             f"- 图片：`{run['image']}`（{run['camera']}）",
             f"- 提示：{', '.join(repr(p) for p in run['prompts'])}；另加一列无文本",
             f"- 框（手工标注，归一化坐标）：" + ('; '.join(f'{n}={b}' for n, b in boxes.items()) or '无'), '',
             '每一格是该文本下，seg（定位头概率）/ cls（CLS 注意力占比）落在各个框里的比例。', '']
    if boxes:
        names = list(boxes)
        lines.append('| 行 | 文本 | ' + ' | '.join(f'seg:{n}' for n in names) + ' | '
                     + ' | '.join(f'cls:{n}' for n in names) + ' |')
        lines.append('|---' * (2 + 2 * len(names)) + '|')
        for row in ROW_ORDER:
            if row not in metrics:
                continue
            for col in metrics[row]['columns']:
                lines.append(f"| {row} | {col['text']} | "
                             + ' | '.join(f"{col['seg_box_mass'][n]:.3f}" for n in names) + ' | '
                             + ' | '.join(f"{col['cls_box_share'][n]:.3f}" for n in names) + ' |')
        lines.append('')
    lines += ['## 换提示后热图变了多少（总变差 TV，0 = 完全相同，1 = 完全不重叠）', '',
              '| 行 | 文本 A | 文本 B | seg TV | cls TV |', '|---|---|---|---|---|']
    for row in ROW_ORDER:
        if row in metrics:
            for p in metrics[row]['pairs']:
                lines.append(f"| {row} | {p['a']} | {p['b']} | {p['seg_tv']:.3f} | {p['cls_tv']:.3f} |")
    lines += ['', '图：`seg_grid.png`（定位头）、`cls_grid.png`（CLS 注意力，即官方 demo 的热图）。', '']
    return '\n'.join(lines)


# ----------------------------------------------------------------------------- model side
class ClsAttention:
    """Last-block CLS->patch attention exactly as TCAttentionExtract reads it
    (attention_map[:, :, cls, patch_start:], mean over heads). Read-only qkv hook,
    same math as visualize_steer_tokens.RegisterAttention."""

    def __init__(self, attn, cls_index, patch_start):
        self.attn, self.cls, self.start = attn, cls_index, patch_start
        self.value = None
        self.handle = attn.qkv.register_forward_hook(self.capture)

    def capture(self, module, inputs, output):
        import torch
        b, n, _ = output.shape
        a = self.attn
        with torch.autocast(device_type=output.device.type, enabled=False):
            q, k, _ = output.float().reshape(b, n, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
            q, k = a.q_norm(q), a.k_norm(k)
            w = ((q[:, :, self.cls:self.cls + 1] * a.scale) @ k.transpose(-2, -1)).softmax(-1)
            self.value = w[:, :, 0, self.start:].mean(1).detach().float().cpu()


def cls_postprocess(attn_rows, grid, suppress):
    """TCAttentionExtract.get_attention_heatmaps after the head mean: official sink
    suppression, then min-max to [0, 1]."""
    h = attn_rows.reshape(attn_rows.shape[0], *grid)
    h, _ = suppress(h)
    lo = h.amin(dim=(-2, -1), keepdim=True)
    hi = h.amax(dim=(-2, -1), keepdim=True)
    return ((h - lo) / (hi - lo + 1e-6)).reshape(attn_rows.shape[0], -1).numpy()


def run_official(args, texts, roberta, dinov2_weights, device):
    """Official SteerViT, constructed as SteerViT.from_pretrained does; only the paths are
    redirected to local files (RoBERTa dir, DINOv2 safetensors given to timm the same way
    DrivoR's ImgEncoder does)."""
    import torch
    import timm
    from timm.data import resolve_data_config
    if args.steervit_repo is not None:
        sys.path.insert(0, str(args.steervit_repo.resolve() / 'src'))
    original_create = timm.create_model

    def create_with_local_weights(name, *a, **k):
        if k.get('pretrained'):
            k.setdefault('pretrained_cfg_overlay', dict(file=str(dinov2_weights)))
        return original_create(name, *a, **k)

    timm.create_model = create_with_local_weights
    try:
        from steervit import SteerViT
        from steervit.utils import suppress_attention_sinks_multi
        ckpt = torch.load(str(args.steervit_checkpoint), map_location='cpu', weights_only=False)
        config = copy.deepcopy(ckpt['config'])
        config['text_encoder'] = str(roberta)
        model = SteerViT(config)
    finally:
        timm.create_model = original_create
    state = ckpt['state_dict']
    if state and all(k.startswith('module.') for k in state):
        state = {k[len('module.'):]: v for k, v in state.items()}
    loaded = model.load_state_dict(state, strict=False)          # as SteerViT.from_pretrained
    steer_missing = [k for k in loaded.missing_keys
                     if '.gated_cross_attn.' in k or k.startswith(('connector.', 'lin_seg_head.'))]
    if steer_missing:
        raise RuntimeError(f'Official SteerViT steer weights missing: {steer_missing[:5]}')
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model = model.to(device)

    trunk = model.vision_model.trunk
    res = model.image_size[0]
    grid = (res // model.patch_size, res // model.patch_size)
    data_cfg = resolve_data_config({}, model=trunk)
    mean = torch.tensor(data_cfg['mean'], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(data_cfg['std'], dtype=torch.float32).view(3, 1, 1)
    im = Image.open(args.image).convert('RGB').resize((res, res), Image.BICUBIC)
    x = (torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.).permute(2, 0, 1) - mean) / std
    x = x[None].to(device)

    capture = ClsAttention(trunk.blocks[-1].attn, 0, trunk.num_prefix_tokens)
    seg, cls = [], []
    try:
        with torch.inference_mode():
            for text in texts:
                feats = model(x, [text]) if text else model(x)
                logits = model.get_heatmap_logits(feats).float()
                seg.append(logits.softmax(-1)[0].cpu().numpy())
                cls.append(capture.value)
    finally:
        capture.handle.remove()
    cls = cls_postprocess(torch.cat(cls), grid, suppress_attention_sinks_multi)
    reference = {
        'steer': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                  if '.gated_cross_attn.' in k or k.startswith(('connector.', 'lin_seg_head.'))},
        'trunk': {k: v.detach().cpu().clone() for k, v in trunk.state_dict().items()
                  if '.gated_cross_attn.' not in k},
    }
    info = dict(resolution=res, grid=list(grid), data_config={k: str(v) for k, v in data_cfg.items()},
                missing_keys=len(loaded.missing_keys), unexpected_keys=len(loaded.unexpected_keys),
                tanh_gates={i: float(b.gated_cross_attn.attn_gate.tanh()) for i, b in enumerate(trunk.blocks)
                            if b.gated_cross_attn is not None},
                preprocessing='whole image resized to 336x336 bicubic + official mean/std (no centre crop)')
    row = dict(grid=list(grid), seg=np.stack(seg), cls=cls)
    model.text_model.cpu()
    del model
    torch.cuda.empty_cache()
    return row, info, reference, suppress_attention_sinks_multi


def compare_with_official(encoder, reference):
    """pure row integrity: our steer modules and DINOv2 trunk vs the official model."""
    ours = encoder.state_dict()
    steer_equal, steer_diff = 0, []
    for key, value in reference['steer'].items():
        if key.startswith('vision_model.trunk.'):
            ours_key = 'model.lora_vit.' + key[len('vision_model.trunk.'):] if encoder.use_lora \
                else 'model.' + key[len('vision_model.trunk.'):]
        else:
            ours_key = key
        if ours_key in ours and torch_equal(ours[ours_key], value):
            steer_equal += 1
        else:
            steer_diff.append(key)
    trunk_equal, trunk_diff, skipped = 0, [], []
    prefix = 'model.lora_vit.' if encoder.use_lora else 'model.'
    for key, value in reference['trunk'].items():
        if key == 'pos_embed':
            skipped.append(key)              # 24x24 vs 48x82 grid: resampled from the same weights
            continue
        candidates = [prefix + key]
        if key.endswith(('attn.qkv.weight', 'attn.qkv.bias')):
            candidates.insert(0, prefix + key.replace('attn.qkv.', 'attn.qkv.qkv.'))
        match = next((c for c in candidates if c in ours), None)
        if match is not None and torch_equal(ours[match], value):
            trunk_equal += 1
        else:
            trunk_diff.append(key)
    return dict(steer_tensors_equal=steer_equal, steer_tensors_different=steer_diff,
                trunk_tensors_equal=trunk_equal, trunk_tensors_different=trunk_diff[:20],
                trunk_tensors_different_count=len(trunk_diff), trunk_skipped=skipped)


def torch_equal(a, b):
    import torch
    return a.shape == b.shape and torch.equal(a.detach().cpu(), b.detach().cpu())


def run_ours(model, encoder, args, texts, cam_index, image, suppress):
    import torch
    start = encoder.num_prefix_tokens + encoder.vit.num_prefix_tokens
    capture = ClsAttention(encoder.vit.blocks[-1].attn, encoder.num_prefix_tokens, start)
    seg, cls = [], []
    try:
        with torch.inference_mode():
            for text in texts:
                logits = model.ground_rows(image, cam_index, [text]).float()
                seg.append(logits.softmax(-1)[0].cpu().numpy())
                cls.append(capture.value)
    finally:
        capture.handle.remove()
    grid = tuple(encoder.patch_grid)
    if seg[0].shape[0] != grid[0] * grid[1]:
        raise RuntimeError(f'{seg[0].shape[0]} patches for grid {grid}')
    return dict(grid=list(grid), seg=np.stack(seg), cls=cls_postprocess(torch.cat(cls), grid, suppress),
                ca_on_registers=encoder.ca_on_registers, gate_scale=float(model.steer_gate_scale))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage2-checkpoint', type=Path, required=True,
                        help='stage-2 .ckpt; its run config gives the model, D and the stage-1 checkpoint')
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--camera', default='cam_f0')
    parser.add_argument('--prompts', nargs='+', required=True)
    parser.add_argument('--box', type=parse_box, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steervit-checkpoint', type=Path, default=None,
                        help='official steervit_dinov2_base.pth (default: init_args of the stage-2 run)')
    parser.add_argument('--steervit-repo', type=Path, default=None,
                        help='official SteerViT repo root (uses <repo>/src); omit if steervit is pip-installed')
    parser.add_argument('--drivor-checkpoint', type=Path, default=None, help='default: init_args of the stage-2 run')
    parser.add_argument('--stage1-checkpoint', type=Path, default=None, help='default: init_args of the stage-2 run')
    parser.add_argument('--roberta', type=Path, default=None, help='default: steer.text_encoder of the stage-2 run')
    parser.add_argument('--rows', nargs='+', default=list(ROW_ORDER), choices=ROW_ORDER)
    args = parser.parse_args()
    boxes = dict(args.box)
    if len(boxes) != len(args.box):
        raise SystemExit('duplicate --box names')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    import torch
    import yaml
    from omegaconf import OmegaConf
    from navsim.agents.drivoR_steer.checkpoint_utils import (load_drivor_checkpoint, load_stage1_checkpoint,
                                                             load_steervit_checkpoint)
    from navsim.agents.drivoR_steer.config_utils import cfg_value
    from navsim.agents.drivoR_steer.steer_data import preprocess_camera_image
    from navsim.agents.drivoR_steer.steer_model import DrivoRSteerModel
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA/PPU unavailable')
    device = torch.device('cuda')
    torch.manual_seed(2)
    np.random.seed(2)

    ckpt2_path = args.stage2_checkpoint.resolve()
    config_path = ckpt2_path.parents[3] / 'code/hydra/config.yaml'
    loader = getattr(yaml, 'CSafeLoader', yaml.SafeLoader)
    saved = yaml.load(config_path.read_text(), Loader=loader)['agent']
    config = saved['config']
    init_args = saved.get('init_args') or {}

    def pick(cli, key):
        value = cli if cli is not None else (init_args.get(key) or None)
        if value is None:
            raise SystemExit(f'--{key.replace("_", "-")} not given and init_args.{key} empty in {config_path}')
        value = Path(value)
        if not value.is_file():
            raise SystemExit(f'{key}: file not found: {value}')
        return value

    steervit_path = pick(args.steervit_checkpoint, 'steervit_checkpoint')
    args.steervit_checkpoint = steervit_path
    need_chain = any(r in args.rows for r in ('init', 'stage1', 'stage2'))
    drivor_path = pick(args.drivor_checkpoint, 'drivor_checkpoint') if need_chain else None
    stage1_path = pick(args.stage1_checkpoint, 'stage1_checkpoint') if \
        any(r in args.rows for r in ('stage1', 'stage2')) else None
    roberta = args.roberta or Path(config['steer']['text_encoder'])
    config['steer']['text_encoder'] = str(roberta)    # same RoBERTa for the official model and ours
    dinov2_weights = Path(config['image_backbone']['model_weights'])
    texts = [''] + list(args.prompts)
    labels = ['(no text)'] + list(args.prompts)

    run = dict(complete=False, image=str(args.image), camera=args.camera, prompts=list(args.prompts),
               boxes=boxes, rows=list(args.rows), stage2_checkpoint=str(ckpt2_path), saved_config=str(config_path),
               steervit_checkpoint=str(steervit_path), drivor_checkpoint=str(drivor_path) if drivor_path else None,
               stage1_checkpoint=str(stage1_path) if stage1_path else None, roberta=str(roberta),
               dinov2_weights=str(dinov2_weights), precision='fp32', seed=2,
               device=torch.cuda.get_device_name(0), row_info={})
    rows = {}

    # 1) official SteerViT (always run: its sink-suppression function is used for every row)
    official_row, official_info, reference, suppress = run_official(args, texts, roberta, dinov2_weights, device)
    if 'official' in args.rows:
        rows['official'] = official_row
        run['row_info']['official'] = official_info
    print('official done', flush=True)

    # 2) our encoder, built from the stage-2 run config
    model = DrivoRSteerModel(OmegaConf.create(config)).eval()
    encoder = model.image_backbone
    stage2_mode = encoder.ca_on_registers             # from the stage-2 config
    chain_mode = True
    if stage1_path is not None:
        meta = torch.load(str(stage1_path), map_location='cpu', mmap=True, weights_only=False).get('steer_meta') or {}
        chain_mode = bool(meta.get('ca_on_registers', True))  # missing = trained before plan A = true
    model = model.to(device)
    image = preprocess_camera_image(args.image, config['image_size'])[None].to(device)
    if model.steer_cameras.count(args.camera) != 1:
        raise SystemExit(f'camera {args.camera} not in {model.steer_cameras}')
    cam_index = torch.tensor([model.steer_cameras.index(args.camera)])

    report = load_steervit_checkpoint(encoder, steervit_path)
    lora_b = {n: float(p.detach().abs().max()) for n, p in model.named_parameters() if '.linear_b_' in n}
    if not lora_b or max(lora_b.values()) != 0.0:
        raise RuntimeError('pure row expects freshly constructed LoRA with B = 0')
    encoder.vit.steer_ca_on_registers = chain_mode
    model.steer_gate_scale.fill_(1.0)
    if 'pure' in args.rows:
        rows['pure'] = run_ours(model, encoder, args, texts, cam_index, image, suppress)
        run['row_info']['pure'] = dict(steervit=report, lora_b_tensors=len(lora_b), lora_b_max=0.0,
                                       same_as_official=compare_with_official(encoder, reference),
                                       ca_on_registers=chain_mode, gate_scale=1.0)
        print('pure done', flush=True)

    if need_chain:
        report = load_drivor_checkpoint(model, drivor_path)
        encoder.vit.steer_ca_on_registers = chain_mode
        model.steer_gate_scale.fill_(1.0)
        if 'init' in args.rows:
            rows['init'] = run_ours(model, encoder, args, texts, cam_index, image, suppress)
            run['row_info']['init'] = dict(drivor=report, ca_on_registers=chain_mode, gate_scale=1.0)
            print('init done', flush=True)

    if stage1_path is not None:
        report = load_stage1_checkpoint(encoder, stage1_path, base_model=model)  # also proves stage 1 used this D
        encoder.vit.steer_ca_on_registers = chain_mode
        model.steer_gate_scale.fill_(1.0)
        if 'stage1' in args.rows:
            rows['stage1'] = run_ours(model, encoder, args, texts, cam_index, image, suppress)
            run['row_info']['stage1'] = dict(stage1=report, ca_on_registers=chain_mode, gate_scale=1.0)
            print('stage1 done', flush=True)

    if 'stage2' in args.rows:
        ckpt = torch.load(str(ckpt2_path), map_location='cpu', mmap=True, weights_only=False)
        prefix = 'agent._drivor_model.'
        state = ckpt['state_dict']
        if not all(k.startswith(prefix) for k in state):
            raise RuntimeError('unexpected key prefix in the stage-2 checkpoint')
        model.load_state_dict({k[len(prefix):]: v for k, v in state.items()}, strict=True)
        encoder.vit.steer_ca_on_registers = stage2_mode
        rows['stage2'] = run_ours(model, encoder, args, texts, cam_index, image, suppress)
        run['row_info']['stage2'] = dict(epoch=int(ckpt['epoch']), global_step=int(ckpt['global_step']),
                                         strict_load=True, ca_on_registers=stage2_mode,
                                         config_ca_on_registers=bool(cfg_value(OmegaConf.create(config).steer,
                                                                               'ca_on_registers', True)),
                                         gate_scale=float(model.steer_gate_scale))
        del ckpt, state
        print('stage2 done', flush=True)

    for name, row in rows.items():
        for key in ('seg', 'cls'):
            if not np.isfinite(row[key]).all():
                raise RuntimeError(f'non-finite {key} in row {name}')
        if not np.allclose(row['seg'].sum(-1), 1, atol=1e-4):
            raise RuntimeError(f'seg of row {name} does not sum to 1')
    metrics = {name: row_metrics(row, labels, boxes) for name, row in rows.items()}
    np.savez_compressed(args.output / 'maps.npz', labels=np.array(labels),
                        **{f'{n}_{k}': rows[n][k] for n in rows for k in ('seg', 'cls')},
                        **{f'{n}_grid': np.array(rows[n]['grid']) for n in rows})
    render(rows, labels, boxes, args.image, 'seg', args.output / 'seg_grid.png')
    render(rows, labels, boxes, args.image, 'cls', args.output / 'cls_grid.png')
    viz.write_json(args.output / 'metrics.json', metrics)
    run.update(complete=True, elapsed_seconds=time.monotonic() - started,
               outputs=['seg_grid.png', 'cls_grid.png', 'metrics.json', 'maps.npz', 'summary.md'])
    viz.write_json(args.output / 'run.json', run)
    (args.output / 'summary.md').write_text(summary_markdown(run, metrics, labels, boxes), encoding='utf-8')
    shutil.copy2(__file__, args.output / Path(__file__).name)
    print((args.output / 'summary.md').read_text(encoding='utf-8'), flush=True)
    print('DONE', args.output, flush=True)


if __name__ == '__main__':
    main()
