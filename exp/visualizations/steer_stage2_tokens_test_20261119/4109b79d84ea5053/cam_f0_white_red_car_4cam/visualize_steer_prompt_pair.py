"""Run paired captions through the native four-camera image branch; inspect front tokens."""
import argparse
import json
from pathlib import Path
import shutil
import time

import visualize_steer_tokens as viz
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Existing camera visualization directory')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompts', nargs=2, default=['the white car', 'the black car'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    meta = json.loads((args.source / 'metadata.json').read_text())
    rows = json.loads((args.source.parents[1] / 'selected_samples.json').read_text())
    row = next(r for r in rows if r['scene_token'] == meta['scene_token'])
    import torch
    import yaml
    from omegaconf import OmegaConf
    from navsim.agents.drivoR_steer.steer_model import DrivoRSteerModel
    from navsim.agents.drivoR_steer.steer_data import preprocess_camera_image
    torch.set_num_threads(4)
    torch.manual_seed(2)
    np.random.seed(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA/PPU unavailable')
    config_path = args.checkpoint.resolve().parents[3] / 'code/hydra/config.yaml'
    config = yaml.load(config_path.read_text(), Loader=yaml.CSafeLoader)['agent']['config']
    print('Building model and strictly loading complete checkpoint...', flush=True)
    model = DrivoRSteerModel(OmegaConf.create(config)).eval()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    prefix = 'agent._drivor_model.'
    state = checkpoint['state_dict']
    assert all(k.startswith(prefix) for k in state)
    loaded = model.load_state_dict({k[len(prefix):]: v for k, v in state.items()}, strict=True)
    encoder = model.image_backbone
    grid = tuple(encoder.patch_grid)
    start = encoder.num_prefix_tokens + encoder.vit.num_prefix_tokens
    assert encoder.num_prefix_tokens == 16 and start == 17 and grid == (48, 82)
    ci = model.steer_cameras.index(meta['camera'])
    run = dict(complete=False, scene_token=meta['scene_token'], camera=meta['camera'],
               image=meta['image'], prompts=args.prompts, checkpoint=str(args.checkpoint.resolve()),
               epoch=int(checkpoint['epoch']), global_step=int(checkpoint['global_step']),
               strict_load=True, model_keys_loaded=len(state), missing_keys=list(loaded.missing_keys),
               unexpected_keys=list(loaded.unexpected_keys), saved_config=str(config_path),
               seed=2, grid_shape=grid, patch_start_index=start, ca_on_registers=encoder.ca_on_registers,
               gate_scale=float(model.steer_gate_scale), precision='FP16 autocast; text and attention reconstruction FP32',
               inference_batch_size=1, cameras_per_sample=4, camera_order=model.steer_cameras,
               input_images=row['images'], original_texts={c: row['scene_text'][c]['text'] for c in model.steer_cameras},
               scope='Native four-camera image_backbone.forward, B=1 N=4; no downstream planning evaluation',
               pca=False, device=torch.cuda.get_device_name(0))
    del checkpoint, state
    viz.write_json(args.output / 'run.json', run)
    model = model.to('cuda').eval()
    capture = viz.RegisterAttention(encoder)
    image_tensor = torch.stack([preprocess_camera_image(row['images'][c], config['image_size'])
                                for c in model.steer_cameras])[None].to('cuda')
    assert image_tensor.shape == (1, 4, 3, 672, 1148)
    captured_tokens = {}
    def save_tokens(module, inputs, output):
        captured_tokens['value'] = output.detach()
    token_hook = encoder.vit.norm.register_forward_hook(save_tokens)
    results = []
    all_camera_results = []
    run['condition_texts'] = []
    try:
        for prompt in args.prompts:
            texts = [prompt if c == meta['camera'] else row['scene_text'][c]['text'] for c in model.steer_cameras]
            run['condition_texts'].append(dict(zip(model.steer_cameras, texts)))
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
                ctx = model.encode_text(texts, torch.tensor([bool(t.strip()) for t in texts]))
                encoder._steer_ctx = ctx
                try:
                    scene_output = encoder(image_tensor, model.scene_embeds)
                    all_logits = encoder._steer_out['ground_logits'].float()[0]
                    all_tokens = captured_tokens.pop('value')
                finally:
                    encoder._steer_ctx = None
                    encoder._steer_out = None
                assert all_tokens.shape == (4, 3953, 768) and scene_output.shape == (1, 64, 256)
                assert capture.value.shape == (4, 16, 3936)
                tokens = all_tokens[ci:ci+1]
                logits = all_logits[ci:ci+1]
                prob = logits.softmax(-1)
                neck = scene_output.reshape(1, 4, 16, 256)[:, ci]
                all_camera_results.append(dict(patches=all_tokens[:, start:].half().cpu().numpy(),
                    scene=all_tokens[:, :16].half().cpu().numpy(),
                    neck=scene_output.reshape(4, 16, 256).half().cpu().numpy(),
                    logits=all_logits.cpu().numpy(), prob=all_logits.softmax(-1).cpu().numpy(),
                    attn=capture.value.copy()))
            assert tokens.shape == (1, 3953, 768)
            results.append(dict(patches=tokens[0, start:].half().cpu().numpy(),
                                scene=tokens[0, :16].half().cpu().numpy(),
                                neck=neck[0].half().cpu().numpy(), logits=logits[0].cpu().numpy(),
                                prob=prob[0].cpu().numpy(), attn=capture.value[ci].copy()))
            print('Four-camera inference complete:', texts, flush=True)
    finally:
        capture.handle.remove()
        token_hook.remove()
    arrays = {key: np.stack([r[key] for r in results]) for key in results[0]}
    for key, value in arrays.items():
        assert np.isfinite(value).all(), key
    probs, attention = arrays['prob'], arrays['attn']
    assert np.allclose(probs.sum(-1), 1, atol=1e-5)
    reg_dist = attention / np.maximum(attention.sum(-1, keepdims=True), 1e-20)
    means = reg_dist.mean(1)
    np.savez_compressed(args.output / 'features.npz', prompts=np.array(args.prompts),
                        patch_features=arrays['patches'], scene_features=arrays['scene'],
                        scene_features_after_neck=arrays['neck'], register_to_patch_attention=attention,
                        ground_probabilities=probs, ground_logits=arrays['logits'], grid_shape=np.array(grid))
    all_arrays = {key: np.stack([r[key] for r in all_camera_results]) for key in results[0]}
    assert all(np.isfinite(v).all() for v in all_arrays.values())
    np.savez_compressed(args.output / 'four_camera_features.npz', camera_order=np.array(model.steer_cameras),
                        prompts=np.array(args.prompts), **all_arrays)
    run['distribution_comparison'] = dict(
        patch_grounding_total_variation=float(np.abs(probs[0] - probs[1]).sum() / 2),
        mean_register_attention_total_variation=float(np.abs(means[0] - means[1]).sum() / 2),
        per_register_attention_total_variation=(np.abs(reg_dist[0] - reg_dist[1]).sum(-1) / 2).tolist(),
        patch_peak_row_col=[list(map(int, np.unravel_index(p.argmax(), grid))) for p in probs])
    model.cpu()
    model._text_encoder.model.cpu()
    torch.cuda.empty_cache()
    im = Image.open(meta['image']).convert('RGB').resize((1148, 672))
    im.save(args.output / 'image.jpg', quality=95)
    image = np.asarray(im)
    pmax, rmax = float(np.percentile(probs, 99.5)), float(np.percentile(means, 99.5))
    reg_max = np.percentile(reg_dist, 99.5, axis=(0, 2))
    run['display'] = dict(patch_vmax=pmax, register_mean_vmax=rmax,
                          per_register_vmax=reg_max.tolist(), percentile=99.5, alpha=.55,
                          normalization='Shared between both prompts for each map type/token; patch-only conditional register attention')
    files = []
    with PdfPages(args.output / 'prompt_comparison.pdf') as pdf:
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        for i, prompt in enumerate(args.prompts):
            axes[0, i].imshow(image)
            axes[0, i].set_title(f'Input: "{prompt}"')
            axes[0, i].axis('off')
            viz.overlay(axes[1, i], image, probs[i].reshape(grid), pmax, 'PATCH grounding (spatial softmax)')
            viz.overlay(axes[2, i], image, means[i].reshape(grid), rmax, 'Mean of 16 REGISTER attention maps')
        fig.suptitle(f"{meta['scene_token']} / {meta['camera']} / four-camera input, front color-word intervention", fontsize=14)
        fig.text(.5, .012, 'Shared color scale between prompts in each row; 99.5th percentile display clipping.\n'
                 'Patch grounding and register attention measure different quantities. No color-specific ground truth supplied.', ha='center')
        fig.tight_layout(rect=(0, .055, 1, .96))
        fig.savefig(args.output / 'comparison.png', dpi=140)
        pdf.savefig(fig)
        plt.close(fig)
        files.append('comparison.png')
        for i, prompt in enumerate(args.prompts):
            fig, axes = plt.subplots(4, 4, figsize=(17, 12))
            for j, ax in enumerate(axes.flat):
                viz.overlay(ax, image, reg_dist[i, j].reshape(grid), float(reg_max[j]), f'Register token {j:02d}')
            fig.suptitle(f'{meta["scene_token"]} / {meta["camera"]} / "{prompt}"\nFinal block; 12 heads averaged; same-token scales shared between prompts')
            fig.tight_layout(rect=(0, .015, 1, .94), h_pad=2.)
            name = f'register_tokens_prompt_{i + 1}.png'
            fig.savefig(args.output / name, dpi=140)
            pdf.savefig(fig)
            plt.close(fig)
            files.append(name)
        # Each PDF page pairs four register IDs to allow direct prompt comparison.
        for first in range(0, 16, 4):
            fig, axes = plt.subplots(4, 2, figsize=(16, 19))
            for j in range(first, first + 4):
                for i, prompt in enumerate(args.prompts):
                    viz.overlay(axes[j-first, i], image, reg_dist[i, j].reshape(grid), float(reg_max[j]),
                                f'Register {j:02d}: "{prompt}"')
            fig.suptitle('Register attention: same token, same scale, different color word')
            fig.tight_layout(rect=(0, .01, 1, .97))
            pdf.savefig(fig)
            name = f'register_pair_{first:02d}_{first+3:02d}.png'
            fig.savefig(args.output / name, dpi=120)
            files.append(name)
            plt.close(fig)
    for name in files:
        with Image.open(args.output / name) as img:
            img.verify()
    run.update(complete=True, elapsed_seconds=time.monotonic()-started,
               validation=dict(finite_arrays=True, probabilities_sum_to_one=True, image_files_decode=True),
               output_images=files, pdf_pages=7)
    viz.write_json(args.output / 'run.json', run)
    import html
    pair_title = html.escape(' / '.join(args.prompts))
    page = f'<!doctype html><meta charset="utf-8"><title>{pair_title}</title><style>body{{max-width:1500px;margin:30px auto;font:18px sans-serif}}img{{width:100%}}</style>'
    page += f'<h1>同图颜色词对照：{pair_title}</h1><p>完整输入四相机图片；只替换 cam_f0 语言，其他相机保留原始语言。展示 cam_f0 的 patch grounding 与最后一层 register attention；相同色标；无 PCA。</p><a href="prompt_comparison.pdf">7 页 PDF</a> · <a href="run.json">运行记录</a>'
    page += ''.join(f'<h2>{html.escape(name)}</h2><a href="{name}"><img src="{name}"></a>' for name in files)
    (args.output / 'index.html').write_text(page)
    shutil.copy2(__file__, args.output / Path(__file__).name)
    print(json.dumps(run['distribution_comparison'], indent=2), flush=True)
    print('DONE', args.output, flush=True)


if __name__ == '__main__':
    main()
