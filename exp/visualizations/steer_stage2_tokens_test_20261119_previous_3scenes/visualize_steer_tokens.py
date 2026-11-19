"""Inspect Stage-2 test images: 16 scene-token attention maps and output patch features.

Uses saved training config and a strict full-model checkpoint load. Runs the native
per-camera encoder path in eval mode; no planner/trajectory scoring is needed.
The final-layer register->patch extraction follows attention_maps_viz.ipynb,
but derives the patch offset and records only 16 query rows (not N x N matrices).
"""
import argparse
import csv
import html
import json
import os
from pathlib import Path
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "nuplan-devkit")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/drivor_token_viz_mpl")

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from navsim.agents.drivoR_steer.steer_model import DrivoRSteerModel
from navsim.agents.drivoR_steer.steer_data import DATA_CAMERAS, load_grounding_npz, preprocess_camera_image

MODES = ("correct", "empty", "shuffled")
# Different logs, chosen by caption categories before inference, not by model results.
DEFAULT_TOKENS = ("6233d1b3e62a5772", "93b50a6df30b5453", "3b56d2b022d25026")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


class RegisterAttention:
    """Read-only QKV hook, with the trained LoRA already included in Q and V.

    Softmax uses ALL keys, then heads are averaged and patch columns extracted.
    FP32 reconstruction avoids low-precision visualization underflow; the native
    attention forward and model outputs are not replaced or changed.
    """
    def __init__(self, encoder):
        self.attn = encoder.vit.blocks[-1].attn
        self.registers = encoder.num_prefix_tokens
        self.start = self.registers + encoder.vit.num_prefix_tokens
        self.value = None
        self.handle = self.attn.qkv.register_forward_hook(self.capture)

    def capture(self, module, inputs, output):
        b, n, _ = output.shape
        a = self.attn
        with torch.autocast(device_type=output.device.type, enabled=False):
            q, k, _ = output.float().reshape(b, n, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
            q, k = a.q_norm(q), a.k_norm(k)
            weights = ((q[:, :, :self.registers] * a.scale) @ k.transpose(-2, -1)).softmax(-1)
            self.value = weights.mean(1)[..., self.start:].detach().cpu().numpy()[0]


def metric(prob, target, valid):
    if not valid or not np.any(target > 0):
        return {"target_mass": None, "peak_in_target": None, "soft_target_ce": None}
    mask = target > 0
    normalized = target / target.sum()
    return {"target_mass": float(prob[mask].sum()),
            "peak_in_target": bool(mask[np.argmax(prob)]),
            "soft_target_ce": float(-(normalized * np.log(np.maximum(prob, 1e-30))).sum())}


def overlay(ax, image, values, vmax, title, cmap="magma"):
    ax.imshow(image)
    ax.imshow(values, extent=(0, image.shape[1], image.shape[0], 0),
              interpolation="nearest", alpha=0.55, cmap=cmap, vmin=0, vmax=max(vmax, 1e-12))
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def pca_colors(patches, grid):
    # Joint basis and joint percentile scales across language conditions, within a view.
    x = torch.from_numpy(np.concatenate(patches).astype(np.float32))
    mean = x.mean(0)
    torch.manual_seed(2)
    _, _, basis = torch.pca_lowrank(x, q=6, center=True, niter=4)
    basis = basis[:, :3]
    projected = ((x - mean) @ basis).numpy()
    lo, hi = np.percentile(projected, [1, 99], axis=0)
    rgb = np.clip((projected - lo) / np.maximum(hi - lo, 1e-8), 0, 1)
    return rgb.reshape(len(MODES), *grid, 3), basis.numpy(), mean.numpy(), lo, hi


def draw_view(out, row, camera, image, results, target, valid, boxes, grid, donor):
    out.mkdir(parents=True, exist_ok=True)
    image_pil = Image.open(row["images"][camera]).convert("RGB").resize((image.shape[1], image.shape[0]))
    image_pil.save(out / "image.jpg", quality=95)
    probabilities = np.stack([r["prob"] for r in results])
    registers = np.stack([r["attn"] for r in results])
    # Per-register patch-conditional distribution makes spatial comparison possible.
    reg_dist = registers / np.maximum(registers.sum(-1, keepdims=True), 1e-20)
    means = reg_dist.mean(1)
    rgb, basis, center, pca_lo, pca_hi = pca_colors([r["patches"] for r in results], grid)
    pmax = float(np.percentile(probabilities, 99.5))
    rmax = float(np.percentile(means, 99.5))
    records = []
    for i, mode in enumerate(MODES):
        m = metric(probabilities[i], target, valid)
        rm = metric(means[i], target, valid)
        records.append(dict(scene_token=row["scene_token"], camera=camera, mode=mode,
                            text=results[i]["text"], grounding_valid=bool(valid),
                            target_area_fraction=float((target > 0).mean()) if valid else None,
                            **m, register_mean_target_mass=rm["target_mass"],
                            ground_tv_from_correct=float(np.abs(probabilities[i] - probabilities[0]).sum() / 2),
                            register_mean_tv_from_correct=float(np.abs(means[i] - means[0]).sum() / 2),
                            patch_mean_cosine_change_from_correct=float((1 - F.cosine_similarity(
                                torch.from_numpy(results[i]["patches"]).float(),
                                torch.from_numpy(results[0]["patches"]).float(), dim=-1)).mean()),
                            scene_output_relative_l2_from_correct=float(np.linalg.norm(
                                results[i]["neck"].astype(np.float32) - results[0]["neck"].astype(np.float32)) /
                                max(np.linalg.norm(results[0]["neck"].astype(np.float32)), 1e-12))))

    fig, axes = plt.subplots(4, 3, figsize=(18, 15))
    axes[0, 0].imshow(image)
    selected = set(boxes.get("selected_gt_indices", []))
    for obj in boxes.get("objects", []):
        if obj["gt_index"] in selected:
            x1, y1, x2, y2 = obj["bbox"]
            axes[0, 0].add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="lime", linewidth=1.3))
    axes[0, 0].set_title("Input + dataset selected projected boxes")
    axes[0, 0].axis("off")
    overlay(axes[0, 1], image, target.reshape(grid), float(target.max()),
            "Dataset soft target (audit only)" if valid else "No valid target: metrics omitted", "Greens")
    change = np.linalg.norm(results[0]["patches"].astype(np.float32) - results[1]["patches"].astype(np.float32), axis=-1)
    change /= np.maximum(np.linalg.norm(results[1]["patches"].astype(np.float32), axis=-1), 1e-12)
    overlay(axes[0, 2], image, change.reshape(grid), float(np.percentile(change, 99.5)),
            "Patch feature relative L2: correct vs empty", "viridis")
    for i, mode in enumerate(MODES):
        mass = records[i]["target_mass"]
        label = f"target mass={mass:.3f}" if mass is not None else "target unavailable"
        overlay(axes[1, i], image, probabilities[i].reshape(grid), pmax, f"{mode}: PATCH grounding / {label}")
        overlay(axes[2, i], image, means[i].reshape(grid), rmax, f"{mode}: mean of 16 SCENE attention maps")
        axes[3, i].imshow(rgb[i], interpolation="nearest")
        axes[3, i].set_title(f"{mode}: output PATCH features, shared PCA RGB")
        axes[3, i].axis("off")
    caption = row["scene_text"][camera]["text"]
    title = f"{row['scene_token']} / {camera}\nCorrect: {textwrap.fill(caption or '[empty]', 145)}\nShuffled: {textwrap.fill(donor['scene_text'][camera]['text'] or '[empty]', 145)}"
    fig.suptitle(title, fontsize=11)
    fig.text(.5, .01, "Heatmap colors: dark=low, yellow=high; 99.5th percentile clipping, shared scale across conditions within each row.\n"
             "Grounding and attention are different quantities. PCA colors are feature coordinates, not semantic labels. Target masks are automatic audit labels.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .90))
    fig.savefig(out / "comparison.png", dpi=130)
    plt.close(fig)

    # One scale per token shared across all three conditions, preserving differences.
    for i, mode in enumerate(MODES):
        fig, axes = plt.subplots(4, 4, figsize=(17, 11))
        for j, ax in enumerate(axes.flat):
            vmax = float(np.percentile(reg_dist[:, j], 99.5))
            overlay(ax, image, reg_dist[i, j].reshape(grid), vmax, f"Scene token {j:02d}")
        fig.suptitle(f"{row['scene_token']} / {camera} / {mode}\nFinal block, 12 heads averaged; 16 scene queries -> 48 x 82 image patches", fontsize=12)
        fig.tight_layout(rect=(0, .015, 1, .94))
        fig.savefig(out / f"scene_tokens_{mode}.png", dpi=120)
        plt.close(fig)

    np.savez_compressed(out / "features.npz", modes=np.array(MODES),
        patch_features=np.stack([r["patches"] for r in results]),
        scene_features=np.stack([r["scene"] for r in results]),
        scene_features_after_neck=np.stack([r["neck"] for r in results]),
        register_to_patch_attention=registers, ground_probabilities=probabilities,
        ground_logits=np.stack([r["logits"] for r in results]), target=target,
        grid_shape=np.array(grid), pca_basis=basis, pca_center=center,
        pca_low=pca_lo, pca_high=pca_hi, patch_relative_l2_correct_empty=change)
    write_json(out / "metadata.json", dict(scene_token=row["scene_token"], camera=camera,
        image=row["images"][camera], grounding_npz=row["grounding_npz"], boxes_json=row["boxes_json"],
        donor_scene=donor["scene_token"], metrics=records))
    return records


def write_gallery(output, rows, records):
    sections = []
    for row in rows:
        cards = []
        for cam in DATA_CAMERAS:
            rel = f"{row['scene_token']}/{cam}"
            ms = [r for r in records if r['scene_token'] == row['scene_token'] and r['camera'] == cam]
            masses = ' / '.join(f"{r['mode']}: {r['target_mass']:.3f}" if r['target_mass'] is not None else f"{r['mode']}: N/A" for r in ms)
            links = ' · '.join(f'<a href="{rel}/scene_tokens_{mode}.png">16 tokens: {mode}</a>' for mode in MODES)
            cards.append(f'<article><h3>{cam}</h3><p>{html.escape(row["scene_text"][cam]["text"] or "[empty language]")}</p>'
                f'<p>Patch probability mass inside audit target — {masses}</p><p>{links} · <a href="{rel}/features.npz">Raw features</a></p>'
                f'<a href="{rel}/comparison.png"><img loading="lazy" src="{rel}/comparison.png"></a></article>')
        sections.append(f'<section><h2>{row["scene_token"]}</h2><p>{row["log_name"]}</p>{"".join(cards)}</section>')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>SteerViT token inspection</title><style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;background:#f6f7fa;color:#19222e}article{background:white;padding:20px;margin:20px 0;border:1px solid #ddd;border-radius:10px}img{width:100%}a{color:#1858ac}p{line-height:1.6}</style>'
    page += '<h1>Stage-2: scene tokens vs output patch tokens</h1><p>3 test scenes, 4 cameras each. All selected views are shown; selection preceded inference. Correct / empty / shuffled language use identical images and weights. Click an image for full resolution.</p>'
    page += '<p><b>Reading the figures:</b> patch grounding uses the trained linear head + spatial softmax. Scene maps use last-layer self-attention averaged over all heads. PCA shows output feature structure. These maps have different meanings; neither attention alone nor a few examples establish general localization accuracy. Green boxes/targets are automatically generated audit labels and can be incomplete.</p>'
    page += '<p><a href="README.md">Method and results</a> · <a href="metrics.csv">All metrics</a> · <a href="run.json">Checkpoint and load audit</a></p>'
    (output / 'index.html').write_text(page + ''.join(sections) + '</html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--test-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', nargs='+', default=list(DEFAULT_TOKENS))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(2)
    np.random.seed(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA/PPU not available')
    args.output.mkdir(parents=True, exist_ok=False)
    cfg_path = args.checkpoint.parents[3] / 'code/hydra/config.yaml'
    config = yaml.load(cfg_path.read_text(), Loader=yaml.CSafeLoader)['agent']['config']
    assert config['steer']['enabled'] and not config['steer']['ca_on_registers']
    print('Building model from saved configuration...', flush=True)
    model = DrivoRSteerModel(OmegaConf.create(config)).eval()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    state = checkpoint['state_dict']
    prefix = 'agent._drivor_model.'
    assert all(k.startswith(prefix) for k in state)
    incompatible = model.load_state_dict({k[len(prefix):]: v for k, v in state.items()}, strict=True)
    encoder = model.image_backbone
    grid = tuple(encoder.patch_grid)
    start = encoder.num_prefix_tokens + encoder.vit.num_prefix_tokens
    assert encoder.num_prefix_tokens == 16 and start == 17 and grid == (48, 82)
    provenance = dict(checkpoint=str(args.checkpoint), checkpoint_bytes=args.checkpoint.stat().st_size,
        epoch=int(checkpoint['epoch']), global_step=int(checkpoint['global_step']),
        saved_config=str(cfg_path), model_keys_loaded=len(state), strict_load=True,
        missing_keys=list(incompatible.missing_keys), unexpected_keys=list(incompatible.unexpected_keys),
        ca_on_registers=encoder.ca_on_registers, gate_scale=float(model.steer_gate_scale),
        grid_shape=list(grid), patch_start_index=start, scene_tokens_per_camera=16,
        model_name=config['image_backbone']['model_name'], text_encoder=config['steer']['text_encoder'],
        device=torch.cuda.get_device_name(0), precision='FP16 autocast; RoBERTa and attention reconstruction FP32',
        seed=2, selection='Fixed 3 distinct logs and caption categories, chosen before inference; all 4 cameras shown',
        scope='Native per-camera image/text encoder inference; full checkpoint strictly loaded; no planner scoring or training',
        complete=False)
    write_json(args.output / 'run.json', provenance)
    del checkpoint, state
    model = model.to('cuda').eval()
    capture = RegisterAttention(encoder)
    all_rows = {r['scene_token']: r for r in map(json.loads, (args.test_dir / 'dataset_index.jsonl').read_text().splitlines())}
    rows = [all_rows[t] for t in args.tokens]
    assert len(rows) >= 2 and len({r['log_name'] for r in rows}) == len(rows)
    assert model.steer_cameras == list(DATA_CAMERAS)
    write_json(args.output / 'selected_samples.json', rows)
    records = []
    started = time.monotonic()
    for ri, row in enumerate(rows):
        donor = rows[(ri + 1) % len(rows)]
        target, valid = load_grounding_npz(row['grounding_npz'], row['scene_token'])
        boxes = json.loads(Path(row['boxes_json']).read_text())
        for ci, camera in enumerate(model.steer_cameras):
            im = preprocess_camera_image(row['images'][camera], config['image_size'])[None].to('cuda')
            image = np.array(Image.open(row['images'][camera]).convert('RGB').resize(tuple(config['image_size'])))
            texts = [row['scene_text'][camera]['text'], '', donor['scene_text'][camera]['text']]
            results = []
            for mode, text in zip(MODES, texts):
                with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
                    ctx = model.encode_text([text], torch.tensor([bool(text.strip())]))
                    tokens = encoder.encode_rows(im, model.scene_embeds[:, ci], ctx)
                    logits = encoder.ground_logits(tokens).float()
                    prob = logits.softmax(-1)
                    neck = encoder.neck(tokens[:, :16])
                assert tokens.shape == (1, 3953, 768)
                assert torch.isfinite(tokens).all() and torch.isfinite(logits).all()
                assert capture.value.shape == (16, 3936)
                results.append(dict(text=text, prob=prob[0].cpu().numpy(), logits=logits[0].cpu().numpy(),
                    attn=capture.value.copy(), patches=tokens[0, start:].half().cpu().numpy(),
                    scene=tokens[0, :16].half().cpu().numpy(), neck=neck[0].half().cpu().numpy()))
                print(f'{ri+1}/{len(rows)} {camera} {mode}: finite tokens {tuple(tokens.shape)}', flush=True)
            records.extend(draw_view(args.output / row['scene_token'] / camera, row, camera, image,
                                     results, target[ci], valid[ci], boxes['cameras'][camera], grid, donor))
            write_json(args.output / 'metrics.json', records)
    capture.handle.remove()
    with (args.output / 'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {}
    for mode in MODES:
        subset = [r for r in records if r['mode'] == mode and r['grounding_valid'] and r['target_mass'] is not None]
        summary[mode] = dict(valid_views=len(subset), mean_target_mass=float(np.mean([r['target_mass'] for r in subset])),
            peak_hits=sum(r['peak_in_target'] for r in subset),
            mean_register_target_mass=float(np.mean([r['register_mean_target_mass'] for r in subset])))
    provenance.update(complete=True, elapsed_seconds=time.monotonic()-started, summary=summary,
        views=len(records)//3, forward_passes=len(records), peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
    write_json(args.output / 'run.json', provenance)
    write_gallery(args.output, rows, records)
    (args.output / 'README.md').write_text('# Stage-2 token visualization\n\n'
        f'Checkpoint: `{args.checkpoint}`\n\nStrict load: {provenance["model_keys_loaded"]} tensors; epoch {provenance["epoch"]}, step {provenance["global_step"]}.\n\n'
        'Open index.html for all views. Each view contains comparison.png, three scene_tokens_*.png grids, image.jpg, metadata.json and features.npz.\n\n'
        'Scene tokens: last self-attention block, average of all 12 heads, query rows 0..15, patch columns 17..3952. Patch-conditional normalization is used for display and spatial mass metrics; raw full-softmax attention is saved. Hooks only read QKV.\n\n'
        'Patch tokens: final encoder norm, 48 x 82 x 768; grounding uses the checkpoint linear head and spatial softmax. PCA uses a joint basis and color scale across the three text conditions within each view. After-neck scene tokens (16 x 256) are also saved.\n\n'
        'Correct and empty language share the same image. Shuffled language is the same camera caption from the next selected scene (cyclic). Empty language disables CA through the native row_active mask. No GridMask or text dropout in eval mode. Camera rows are independent in this encoder; their own learned scene embeddings are selected.\n\n'
        'Targets are the dataset automatic audit-only projection masks, not independently hand-labeled evidence. Invalid views are displayed but omitted from localization metrics. A mismatched caption can still describe some visible objects; these examples cannot establish dataset-wide accuracy or causal object-level correctness.\n\n'
        'Heatmaps clip at the 99.5th percentile only for display. Each comparison row shares a scale across language conditions; each numbered scene token shares a scale across its three plots. Grounding and attention are distinct quantities, so colors across different rows are not directly comparable. All metrics use unclipped probabilities.\n\n'
        'Summary (valid views only):\n\n```json\n' + json.dumps(summary, indent=2) + '\n```\n')
    print(json.dumps(provenance, indent=2), flush=True)
    print('DONE', args.output / 'index.html', flush=True)


if __name__ == '__main__':
    main()
