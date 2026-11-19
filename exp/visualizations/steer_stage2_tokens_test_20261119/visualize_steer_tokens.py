"""Test-only scene-token / patch-grounding inspection with unrelated-language controls.

Strictly loads the full Stage-2 checkpoint and uses its native per-camera encoder.
Last-block register-to-patch maps follow attention_maps_viz.ipynb, with the actual
patch offset (16 registers + 1 CLS). Read-only QKV hooks do not change the forward.
Rendering is performed in separate CPU processes. No PCA is computed or plotted.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import html
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'nuplan-devkit')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/drivor_token_viz_mpl')

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages

CAMERAS = ('cam_f0', 'cam_b0', 'cam_l0', 'cam_r0')
MODES = ('correct', 'empty', 'unrelated')
ORIGINAL_TOKENS = ('6233d1b3e62a5772', '93b50a6df30b5453', '3b56d2b022d25026')
# Intentionally out-of-domain captions: no road, car, cyclist, pedestrian, lane,
# traffic-light, sign, or cone descriptions. They are fixed BEFORE inference.
UNRELATED_TEXTS = (
    'A giant purple dragon breathes green fire above an ice castle on a floating island.',
    'Three astronauts play chess on the surface of Jupiter beneath two glowing moons.',
    'A mermaid with golden hair opens a sunken pirate treasure chest deep underwater.',
    'A wizard levitates a crystal teapot inside a dungeon filled with glowing mushrooms.',
    'An enormous pink octopus knits a wool sweater inside a submarine kitchen.',
    'A winged unicorn sleeps on a cloud made of candy beside a rainbow waterfall.',
    'A microscopic kingdom of dancing bacteria surrounds a golden throne inside a soap bubble.',
    'A royal penguin conducts an orchestra of singing jellyfish on an iceberg in outer space.',
    'A silver phoenix hatches from a ruby egg in the heart of an active volcano.',
    'A dinosaur wearing a crown eats a mountain of chocolate cake inside a marble palace.',
    'A tiny elf rides a flying mushroom through a galaxy of sparkling diamonds.',
    'A magical whale floats above a lavender ocean while carrying a castle of glass.',
)


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def select_samples(test_dir, count, seed):
    entries = [json.loads(s) for s in (test_dir / 'dataset_index.jsonl').read_text().splitlines() if s]
    by_token = {r['scene_token']: r for r in entries}
    if count < len(ORIGINAL_TOKENS):
        raise ValueError('scene-count must be >= 3 (includes the original three scenes)')
    rows = [by_token[t] for t in ORIGINAL_TOKENS]
    used = {r['log_name'] for r in rows}
    grouped = {}
    for r in entries:
        if r['log_name'] not in used:
            grouped.setdefault(r['log_name'], []).append(r)
    rng = random.Random(seed)
    logs = rng.sample(sorted(grouped), count-len(rows))
    rows += [rng.choice(sorted(grouped[log], key=lambda r:r['scene_token'])) for log in logs]
    assert len(rows) == len({r['log_name'] for r in rows}) == count
    for r in rows:
        assert r['partition'] == 'test' and r['camera_order'] == list(CAMERAS)
        for path in list(r['images'].values()) + [r['grounding_npz'], r['boxes_json']]:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    return rows


def controls(rows):
    return {r['scene_token']: {c: UNRELATED_TEXTS[(i*4+j) % len(UNRELATED_TEXTS)]
            for j,c in enumerate(CAMERAS)} for i,r in enumerate(rows)}


class RegisterAttention:
    def __init__(self, encoder):
        self.attn = encoder.vit.blocks[-1].attn
        self.registers = encoder.num_prefix_tokens
        self.start = self.registers + encoder.vit.num_prefix_tokens
        self.value = None
        self.handle = self.attn.qkv.register_forward_hook(self.capture)

    def capture(self, module, inputs, output):
        import torch
        b, n, _ = output.shape
        a = self.attn
        with torch.autocast(device_type=output.device.type, enabled=False):
            q, k, _ = output.float().reshape(b,n,3,a.num_heads,a.head_dim).permute(2,0,3,1,4).unbind(0)
            q, k = a.q_norm(q), a.k_norm(k)
            weights = ((q[:,:,:self.registers] * a.scale) @ k.transpose(-2,-1)).softmax(-1)
            self.value = weights.mean(1)[...,self.start:].detach().cpu().numpy()


def metric(prob, target, valid):
    if not valid or not np.any(target > 0):
        return {'target_mass':None, 'peak_in_target':None, 'soft_target_ce':None}
    mask = target > 0
    return {'target_mass':float(prob[mask].sum()), 'peak_in_target':bool(mask[np.argmax(prob)]),
            'soft_target_ce':float(-((target/target.sum())*np.log(np.maximum(prob,1e-30))).sum())}


def overlay(ax, image, values, vmax, title, cmap='magma'):
    ax.imshow(image)
    ax.imshow(values, extent=(0,image.shape[1],image.shape[0],0), interpolation='nearest',
              alpha=.55, cmap=cmap, vmin=0, vmax=max(vmax,1e-12))
    ax.set_title(title, fontsize=10, pad=7)
    ax.axis('off')


def render_view(out_string, row, camera, results, target, dataset_valid, boxes, grid):
    out = Path(out_string)
    out.mkdir(parents=True, exist_ok=True)
    image_pil = Image.open(row['images'][camera]).convert('RGB').resize((1148,672))
    image_pil.save(out/'image.jpg',quality=95)
    image = np.asarray(image_pil)
    probabilities = np.stack([r['prob'] for r in results])
    registers = np.stack([r['attn'] for r in results])
    reg_dist = registers / np.maximum(registers.sum(-1,keepdims=True),1e-20)
    means = reg_dist.mean(1)
    eligible = bool(dataset_valid and results[0]['text'].strip() and np.any(target > 0))
    records = []
    correct_patch = results[0]['patches'].astype(np.float32)
    for i, mode in enumerate(MODES):
        patch = results[i]['patches'].astype(np.float32)
        cosine = np.sum(correct_patch*patch,axis=-1) / np.maximum(
            np.linalg.norm(correct_patch,axis=-1)*np.linalg.norm(patch,axis=-1),1e-12)
        neck = results[i]['neck'].astype(np.float32)
        base_neck = results[0]['neck'].astype(np.float32)
        records.append(dict(scene_token=row['scene_token'],log_name=row['log_name'],camera=camera,mode=mode,
            text=results[i]['text'],dataset_grounding_valid=bool(dataset_valid),grounding_valid=eligible,
            target_area_fraction=float((target>0).mean()) if eligible else None,
            **metric(probabilities[i],target,eligible),
            register_mean_target_mass=metric(means[i],target,eligible)['target_mass'],
            ground_tv_from_correct=float(np.abs(probabilities[i]-probabilities[0]).sum()/2),
            register_mean_tv_from_correct=float(np.abs(means[i]-means[0]).sum()/2),
            ground_entropy_normalized=float(-(probabilities[i]*np.log(np.maximum(probabilities[i],1e-30))).sum()/np.log(probabilities.shape[-1])),
            ground_peak_probability=float(probabilities[i].max()),
            patch_mean_cosine_change_from_correct=float(np.clip(1-cosine,0,2).mean()) if i else 0.,
            scene_output_relative_l2_from_correct=float(np.linalg.norm(neck-base_neck)/max(np.linalg.norm(base_neck),1e-12))))
    change = np.linalg.norm(correct_patch-results[1]['patches'].astype(np.float32),axis=-1)
    change /= np.maximum(np.linalg.norm(results[1]['patches'].astype(np.float32),axis=-1),1e-12)
    fig, axes = plt.subplots(3,3,figsize=(18,12))
    axes[0,0].imshow(image)
    selected = set(boxes.get('selected_gt_indices',[]))
    for obj in boxes.get('objects',[]):
        if obj['gt_index'] in selected:
            x1,y1,x2,y2 = obj['bbox']
            axes[0,0].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='lime',linewidth=1.3))
    axes[0,0].set_title('Input + dataset selected projected boxes')
    axes[0,0].axis('off')
    overlay(axes[0,1],image,target.reshape(grid),float(target.max()),
            'Dataset target (audit only)' if eligible else 'No valid target/text: metrics omitted','Greens')
    overlay(axes[0,2],image,change.reshape(grid),float(np.percentile(change,99.5)),
            'Patch feature relative L2: correct vs empty','viridis')
    pmax,rmax = float(np.percentile(probabilities,99.5)),float(np.percentile(means,99.5))
    for i,mode in enumerate(MODES):
        mass = records[i]['target_mass']
        label = f'target mass={mass:.3f}' if mass is not None else 'target unavailable'
        overlay(axes[1,i],image,probabilities[i].reshape(grid),pmax,f'{mode}: PATCH grounding / {label}')
        overlay(axes[2,i],image,means[i].reshape(grid),rmax,f'{mode}: mean of 16 SCENE attention maps')
    fig.suptitle(f"{row['scene_token']} / {camera}\nCorrect: {textwrap.fill(results[0]['text'] or '[empty]',145)}\nUnrelated: {textwrap.fill(results[2]['text'],145)}",fontsize=11)
    fig.text(.5,.012,'Dark=low, yellow=high. Display clipping: 99.5th percentile, shared across conditions within each heatmap row.\n'
             'Grounding-head predictions and scene-token attention have different meanings. Targets are automatic audit labels.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.87),h_pad=1.8)
    fig.savefig(out/'comparison.png',dpi=130)
    plt.close(fig)
    for i,mode in enumerate(MODES):
        fig,axes = plt.subplots(4,4,figsize=(17,12))
        for j,ax in enumerate(axes.flat):
            overlay(ax,image,reg_dist[i,j].reshape(grid),float(np.percentile(reg_dist[:,j],99.5)),f'Scene token {j:02d}')
        fig.suptitle(f"{row['scene_token']} / {camera} / {mode}\nFinal block, 12 heads averaged; 16 scene queries -> 48 x 82 patches",fontsize=12)
        fig.tight_layout(rect=(0,.015,1,.94),h_pad=2.)
        fig.savefig(out/f'scene_tokens_{mode}.png',dpi=120)
        plt.close(fig)
    np.savez_compressed(out/'features.npz',modes=np.array(MODES),
        patch_features=np.stack([r['patches'] for r in results]),
        scene_features=np.stack([r['scene'] for r in results]),
        scene_features_after_neck=np.stack([r['neck'] for r in results]),
        register_to_patch_attention=registers,ground_probabilities=probabilities,
        ground_logits=np.stack([r['logits'] for r in results]),target=target,
        grid_shape=np.array(grid),patch_relative_l2_correct_empty=change)
    write_json(out/'metadata.json',dict(scene_token=row['scene_token'],camera=camera,image=row['images'][camera],
        grounding_npz=row['grounding_npz'],boxes_json=row['boxes_json'],metrics=records))
    return records


def summarize(records, seed):
    result = {}
    for mode in MODES:
        subset = [r for r in records if r['mode']==mode and r['grounding_valid']]
        by_scene = {}
        for r in subset:
            by_scene.setdefault(r['scene_token'],[]).append(r['target_mass'])
        result[mode] = dict(valid_views=len(subset),valid_scenes=len(by_scene),
            mean_target_mass=float(np.mean([r['target_mass'] for r in subset])),
            mean_per_scene_target_mass=float(np.mean([np.mean(v) for v in by_scene.values()])),
            peak_hits=sum(r['peak_in_target'] for r in subset),
            mean_register_target_mass=float(np.mean([r['register_mean_target_mass'] for r in subset])),
            mean_entropy_normalized=float(np.mean([r['ground_entropy_normalized'] for r in subset])),
            mean_ground_tv_from_correct=float(np.mean([r['ground_tv_from_correct'] for r in subset])))
    groups = {}
    for r in records:
        if r['grounding_valid']:
            groups.setdefault((r['scene_token'],r['camera']),{})[r['mode']]=r
    paired = {}
    rng = np.random.default_rng(seed)
    for control in ('empty','unrelated'):
        diffs = [g['correct']['target_mass']-g[control]['target_mass'] for g in groups.values()]
        per_scene = {}
        for (token,_),g in groups.items():
            per_scene.setdefault(token,[]).append(g['correct']['target_mass']-g[control]['target_mass'])
        macro = np.array([np.mean(v) for v in per_scene.values()])
        boot = macro[rng.integers(0,len(macro),size=(5000,len(macro)))].mean(1)
        paired[control] = dict(better=sum(d>1e-6 for d in diffs),tied=sum(abs(d)<=1e-6 for d in diffs),
            worse=sum(d < -1e-6 for d in diffs),mean_difference=float(np.mean(diffs)),
            mean_per_scene_difference=float(macro.mean()),scene_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist())
    result['paired_correct_minus_control'] = paired
    return result


def table(summary):
    lines = ['| 语言条件 | 有效图片 | 目标区域内平均 patch 概率 | 最高点命中 |',
             '|---|---:|---:|---:|']
    for key,label in [('correct','原始语言'),('empty','空语言'),('unrelated','离谱无关语言')]:
        x=summary[key]
        lines.append(f"| {label} | {x['valid_views']} | {100*x['mean_target_mass']:.2f}% | {x['peak_hits']}/{x['valid_views']} |")
    return '\n'.join(lines)


def conclusions(summary):
    paired=summary['paired_correct_minus_control']
    n=summary['correct']['valid_views']
    return (f"原始语言相对空语言，目标区域内概率平均变化 {paired['empty']['mean_difference']*100:+.2f} 个百分点，"
            f"在 {paired['empty']['better']}/{n} 张有效图片上更高；相对离谱无关语言，平均变化 "
            f"{paired['unrelated']['mean_difference']*100:+.2f} 个百分点，在 {paired['unrelated']['better']}/{n} 张上更高。\n\n"
            '这些统计描述的是定位分布随语言改变的情况，不能单独证明逐词语义理解或规划收益。'
            '尤其无关文字并不描述图中真实目标；其“目标区域概率”仍使用原始文字对应的参考 mask，衡量的是保留了多少原目标响应。'
            '空间 softmax 必须把概率分配到某些 patch，当前头没有“没有匹配对象”输出，因此错误语言下仍出现热区不等于模型认为文字正确。'
            '离谱文字是分布外测试，不能代替同场景、不同真实对象之间的精细语义对照。')


def write_report(output, rows, records, provenance):
    summary=provenance['summary']
    text = f'''# 30 场景扩展可视化（无 PCA）

当前结果：{len(rows)} 个不同日志的 test 场景、{len(rows)*4} 张图片、{len(records)} 个语言条件推理。
checkpoint：`{provenance['checkpoint']}`，epoch {provenance['epoch']}、step {provenance['global_step']}。
完整模型严格加载 {provenance['model_keys_loaded']} 个状态项，缺失和多余项均为 0。

## 更新后的统计

{table(summary)}

{conclusions(summary)}

以上百分比是空间概率质量，不是准确率或 IoU。参考区域来自自动生成的 audit 标签。
全部 {len(rows)*4} 张图片均展示；统计只纳入参考 mask 有效且原始文字非空的图片。
统计表按有效图片等权平均，run.json 另提供按场景等权的结果及按场景重采样的配对差值区间；这些区间仅描述本次选取样本的波动。

## 与上一版的关系

上一版只有 3 个场景，原始/空/其他驾驶场景文字的概率分别为 84.22%/27.47%/78.49%。
本版保留原来 3 个场景，再用 seed={provenance['seed']} 从其余日志随机选择 27 个不同日志、每个日志随机选一个场景；选择发生在推理之前，不按效果筛选。
本版错误输入改为固定的奇幻/太空/海底等无关描述，不再使用其他驾驶场景的文字。
样本和错误输入均改变，因此不能把本版与旧版百分比的差异解释为模型能力发生变化。以上扩展统计替代旧版的小样本结论。

## 怎样读图

打开 index.html；总览 PDF 是 token_comparison.pdf，每个场景还有 8 页 scene_comparison.pdf。
综合对照的第一行：输入及参考框、自动参考 mask、原始语言相对空语言引起的 patch 特征相对 L2 变化。
第二行：最终 patch 特征经过已训练 grounding head 和空间 softmax 后的定位图。
第三行：最后一层 12 个头平均后的 16 个场景 token 自注意力均值。
第二、三行从左到右为原始、空、离谱无关语言。图顶及每个子目录 metadata.json 显示实际文字。
scene_tokens_correct/empty/unrelated.png 分别显示 16 个 token 的逐个注意力。

场景 token 图是最后一层的注意力读出，patch 图是有定位监督的预测头输出，两者含义和监督不同。
不能根据热图更集中就断言 patch 表征更好、16 token 丢失了全部定位信息，或直接使用 patch 一定改善规划。
本次没有规划评分，只有原生逐相机图文编码器推理。

## 复现与数据

图像预处理与工程一致：1148×672，ImageNet 标准化；每个相机使用自己的 scene embeddings。
使用 eval、FP16 autocast，RoBERTa 和可视化注意力重建使用 FP32。无 GridMask/text dropout。
三个条件作为独立相机行组成 batch=3，模型没有行间交互；训练配置及 checkpoint 均不修改。
场景 token 的 query 是 0..15；patch 从 17 起，共 48×82=3936 个。hook 只读 QKV，不改变推理。
注意力先对所有 key 做 softmax，再取 patch 列；显示时按每个场景 token 的 patch 总质量归一化。
热图仅显示时截断到 99.5 分位；同一行的三个条件共用色阶，每个编号 token 也共用三条件色阶。统计使用未截断的概率。

每个图片目录包含原图、4 张可视化、metadata.json、features.npz。
features.npz 保存三个条件的 patch [3,3936,768]、scene [3,16,768]、neck 输出 [3,16,256]（FP16），以及概率、logits、原始注意力等 FP32 数组；没有 PCA 数组。
selected_samples.json、unrelated_prompts.json 和 selection.json 在推理之前固定样本与错误输入。
metrics.csv/metrics.json 保存逐图统计；run.json 保存加载、精度、汇总和自检信息；validation.json 保存产物检查。
'''
    (output/'README.md').write_text(text)
    summary_html = '<table><tr><th>条件</th><th>有效图片</th><th>参考区域概率</th><th>最高点命中</th></tr>'
    for mode,label in [('correct','原始'),('empty','空'),('unrelated','离谱无关')]:
        s=summary[mode]
        summary_html+=f"<tr><td>{label}</td><td>{s['valid_views']}</td><td>{s['mean_target_mass']:.2%}</td><td>{s['peak_hits']}/{s['valid_views']}</td></tr>"
    summary_html+='</table>'
    nav=' '.join(f'<a href="#s{i+1}">{i+1:02d}</a>' for i in range(len(rows)))
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>30 场景 token 对照</title><style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;background:#f6f7fa;color:#19222e}article{background:white;padding:20px;margin:20px 0;border:1px solid #ddd;border-radius:10px}img{width:100%}a{color:#1858ac}p{line-height:1.7}td,th{padding:10px;border:1px solid #ccc}table{border-collapse:collapse}summary{cursor:pointer;font-size:20px;padding:18px;background:#e8edf5}nav{line-height:2.5}nav a{margin:5px}code{overflow-wrap:anywhere}</style>'
    page+=f'<h1>{len(rows)} 场景：16 个场景 token 与 patch 定位对照</h1><p>{len(rows)*4} 张图片；原始 / 空 / 离谱无关语言。没有 PCA。全部样本在推理前选定。</p>'
    page+=summary_html+f'<p>{html.escape(conclusions(summary))}</p>'
    page+='<p><a href="README.md">中文方法与结论</a> · <a href="token_comparison.pdf">全部图片综合对照 PDF</a> · <a href="metrics.csv">逐图统计</a> · <a href="run.json">运行记录</a> · <a href="validation.json">检查记录</a></p><nav>'+nav+'</nav>'
    for i,row in enumerate(rows):
        token=row['scene_token']
        page+=f'<details id="s{i+1}"'+(' open' if i==0 else '')+f'><summary>{i+1:02d} · {token} · {row["log_name"]}</summary><p><a href="{token}/scene_comparison.pdf">本场景 PDF（8 页，含原始语言的 16-token 图）</a></p>'
        for cam in CAMERAS:
            rel=f'{token}/{cam}'
            ms=[r for r in records if r['scene_token']==token and r['camera']==cam]
            wrong=next(r['text'] for r in ms if r['mode']=='unrelated')
            page+=f'<article><h3>{cam}</h3><p>原始：{html.escape(row["scene_text"][cam]["text"] or "[空]")}<br>无关：{html.escape(wrong)}</p>'
            page+='<p>'+' · '.join(f'<a href="{rel}/scene_tokens_{m}.png">16 token：{m}</a>' for m in MODES)+f' · <a href="{rel}/features.npz">原始特征</a></p>'
            page+=f'<a href="{rel}/comparison.png"><img loading="lazy" src="{rel}/comparison.png"></a></article>'
        page+='</details>'
    (output/'index.html').write_text(page+'</html>')


def export_pdfs(output, rows):
    # Stream the large overview instead of retaining hundreds of decoded images.
    with PdfPages(output/'token_comparison.pdf') as pdf:
        for row in rows:
            pages=[]
            for camera in CAMERAS:
                folder=output/row['scene_token']/camera
                with Image.open(folder/'comparison.png') as src:
                    image=src.convert('RGB')
                    pages.append(image.copy())
                    fig=plt.figure(figsize=(12,8))
                    ax=fig.add_axes([0,0,1,1]);ax.imshow(image);ax.axis('off')
                    pdf.savefig(fig,dpi=140);plt.close(fig)
                with Image.open(folder/'scene_tokens_correct.png') as src:
                    pages.append(src.convert('RGB').copy())
            pages[0].save(output/row['scene_token']/'scene_comparison.pdf',save_all=True,
                          append_images=pages[1:],resolution=150,quality=90)
            for page in pages:page.close()
            print('PDF:',row['scene_token'],flush=True)


def validate(output, rows):
    import re
    checked=[]
    for row in rows:
        for camera in CAMERAS:
            folder=output/row['scene_token']/camera
            with np.load(folder/'features.npz',allow_pickle=False) as a:
                assert a['modes'].tolist()==list(MODES)
                assert not any('pca' in k for k in a.files)
                for key,shape in {'patch_features':(3,3936,768),'scene_features':(3,16,768),
                    'scene_features_after_neck':(3,16,256),'register_to_patch_attention':(3,16,3936),
                    'ground_probabilities':(3,3936)}.items():
                    v=a[key];assert v.shape==shape and np.isfinite(v).all(),(folder,key)
                logits=a['ground_logits'];p=np.exp(logits-logits.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
                assert np.allclose(p,a['ground_probabilities'],atol=1e-6)
                assert np.allclose(a['ground_probabilities'].sum(-1),1,atol=1e-5)
            for name in ['comparison.png']+[f'scene_tokens_{m}.png' for m in MODES]:
                with Image.open(folder/name) as im:im.verify()
            checked.append(str(folder.relative_to(output)))
    # validation.json itself is published after this check.
    missing=[x for x in re.findall(r'(?:href|src)="([^"]+)"',(output/'index.html').read_text())
             if not x.startswith('#') and x!='validation.json' and not (output/x).is_file()]
    assert not missing,missing
    report=dict(passed=True,views=len(checked),comparison_images=len(checked),scene_token_grids=len(checked)*3,
        scene_pdfs=len(rows),overview_pdf_pages=len(checked),feature_shapes_finite=True,
        probabilities_sum_to_one=True,saved_logits_match_probabilities=True,no_pca_arrays=True,
        all_image_files_decode=True,gallery_links_exist=True)
    write_json(output/'validation.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--test-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scene-count',type=int,default=30)
    parser.add_argument('--seed',type=int,default=2)
    parser.add_argument('--render-workers',type=int,default=2)
    args=parser.parse_args()
    args.checkpoint=args.checkpoint.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=select_samples(args.test_dir,args.scene_count,args.seed)
    wrong=controls(rows)
    write_json(args.output/'selected_samples.json',rows)
    write_json(args.output/'unrelated_prompts.json',wrong)
    selection=dict(seed=args.seed,scene_count=len(rows),distinct_logs=len(rows),original_tokens=list(ORIGINAL_TOKENS),
        method='Original three scenes plus uniformly sampled distinct remaining logs, one randomly selected test scene per log; selected before inference, no quality filtering',
        wrong_language='Fixed absurd out-of-domain captions, round-robin assignment before inference',
        wrong_prompt_catalog=list(UNRELATED_TEXTS),test_index=str(args.test_dir/'dataset_index.jsonl'))
    write_json(args.output/'selection.json',selection)
    print(f'Selected {len(rows)} scenes, {len(rows)*4} views; all inputs exist.',flush=True)
    import torch
    import yaml
    from omegaconf import OmegaConf
    from navsim.agents.drivoR_steer.steer_model import DrivoRSteerModel
    from navsim.agents.drivoR_steer.steer_data import preprocess_camera_image,load_grounding_npz
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed)
    if not torch.cuda.is_available():raise RuntimeError('CUDA/PPU unavailable')
    config_path=args.checkpoint.parents[3]/'code/hydra/config.yaml'
    config=yaml.load(config_path.read_text(),Loader=yaml.CSafeLoader)['agent']['config']
    assert config['steer']['enabled'] and not config['steer']['ca_on_registers']
    print('Building saved model...',flush=True)
    model=DrivoRSteerModel(OmegaConf.create(config)).eval()
    checkpoint=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    state=checkpoint['state_dict'];prefix='agent._drivor_model.'
    assert all(k.startswith(prefix) for k in state)
    loaded=model.load_state_dict({k[len(prefix):]:v for k,v in state.items()},strict=True)
    encoder=model.image_backbone;grid=tuple(encoder.patch_grid)
    start=encoder.num_prefix_tokens+encoder.vit.num_prefix_tokens
    assert encoder.num_prefix_tokens==16 and start==17 and grid==(48,82)
    assert model.steer_cameras==list(CAMERAS)
    provenance=dict(complete=False,checkpoint=str(args.checkpoint),checkpoint_bytes=args.checkpoint.stat().st_size,
        epoch=int(checkpoint['epoch']),global_step=int(checkpoint['global_step']),saved_config=str(config_path),
        model_keys_loaded=len(state),strict_load=True,missing_keys=list(loaded.missing_keys),unexpected_keys=list(loaded.unexpected_keys),
        ca_on_registers=encoder.ca_on_registers,gate_scale=float(model.steer_gate_scale),
        model_name=config['image_backbone']['model_name'],text_encoder=config['steer']['text_encoder'],
        grid_shape=list(grid),patch_start_index=start,scene_tokens_per_camera=16,
        seed=args.seed,selection=selection,scene_count=len(rows),views=len(rows)*4,conditions=list(MODES),pca=False,
        device=torch.cuda.get_device_name(0),precision='FP16 autocast; RoBERTa and reconstructed attention FP32',
        scope='Native per-camera encoder inference only; full checkpoint strict load; no planner scoring/training',
        inference_batch_size=3,render_workers=args.render_workers)
    write_json(args.output/'run.json',provenance)
    del checkpoint,state
    model=model.to('cuda').eval();capture=RegisterAttention(encoder)
    records=[];pending=[];started=time.monotonic();forward_count=0
    executor=ProcessPoolExecutor(max_workers=args.render_workers,mp_context=multiprocessing.get_context('spawn'))
    def collect(future):
        records.extend(future.result())
        write_json(args.output/'metrics.json',records)
        print(f'Rendered {len(records)//3}/{len(rows)*4} views',flush=True)
    try:
        for ri,row in enumerate(rows):
            target,valid=load_grounding_npz(row['grounding_npz'],row['scene_token'])
            boxes=json.loads(Path(row['boxes_json']).read_text())
            for ci,camera in enumerate(CAMERAS):
                texts=[row['scene_text'][camera]['text'],'',wrong[row['scene_token']][camera]]
                image=preprocess_camera_image(row['images'][camera],config['image_size'])[None].to('cuda')
                with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16):
                    ctx=model.encode_text(texts,torch.tensor([bool(t.strip()) for t in texts]))
                    tokens=encoder.encode_rows(image.expand(3,-1,-1,-1),model.scene_embeds[:,ci].expand(3,-1,-1),ctx)
                    logits=encoder.ground_logits(tokens).float();probs=logits.softmax(-1)
                    neck=encoder.neck(tokens[:,:16])
                assert tokens.shape==(3,3953,768) and capture.value.shape==(3,16,3936)
                assert torch.isfinite(tokens).all() and torch.isfinite(logits).all()
                patch_np=tokens[:,start:].half().cpu().numpy()
                scene_np=tokens[:,:16].half().cpu().numpy();neck_np=neck.half().cpu().numpy()
                logits_np=logits.cpu().numpy();probs_np=probs.cpu().numpy()
                results=[dict(text=t,prob=probs_np[i],logits=logits_np[i],attn=capture.value[i].copy(),
                    patches=patch_np[i],scene=scene_np[i],neck=neck_np[i]) for i,t in enumerate(texts)]
                # Check the changed batching against native single-row inference once.
                if forward_count==0:
                    with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16):
                        single_ctx=model.encode_text([texts[0]],torch.tensor([bool(texts[0].strip())]))
                        single=encoder.encode_rows(image,model.scene_embeds[:,ci],single_ctx)
                        single_prob=encoder.ground_logits(single).float().softmax(-1)[0].cpu().numpy()
                    tv=float(np.abs(single_prob-probs_np[0]).sum()/2)
                    assert tv<.01,('Batch/single distribution mismatch',tv)
                    provenance['batch_vs_single_check']=dict(ground_probability_tv=tv,passed=True,tolerance=.01)
                    print('Batch vs single TV:',tv,flush=True)
                    write_json(args.output/'run.json',provenance)
                forward_count+=3
                pending.append(executor.submit(render_view,str(args.output/row['scene_token']/camera),
                    row,camera,results,target[ci],bool(valid[ci]),boxes['cameras'][camera],grid))
                print(f'Inference {ri+1}/{len(rows)} {camera}: all 3 conditions finite',flush=True)
                if len(pending)>=4:collect(pending.pop(0))
        for future in pending:collect(future)
    finally:
        capture.handle.remove();executor.shutdown(wait=True,cancel_futures=True)
    order={r['scene_token']:i for i,r in enumerate(rows)}
    records.sort(key=lambda r:(order[r['scene_token']],CAMERAS.index(r['camera']),MODES.index(r['mode'])))
    write_json(args.output/'metrics.json',records)
    with (args.output/'metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    provenance.update(summary=summarize(records,args.seed),condition_forwards=forward_count,
        diagnostic_extra_single_forward=1,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
    # Release model device storage while CPU artifact generation finishes.
    model.cpu();model._text_encoder.model.cpu();torch.cuda.empty_cache()
    write_report(args.output,rows,records,provenance)
    export_pdfs(args.output,rows)
    provenance['validation']=validate(args.output,rows)
    provenance.update(complete=True,elapsed_seconds=time.monotonic()-started)
    write_json(args.output/'run.json',provenance)
    for name in ['visualize_steer_tokens.py','run_steer_tokens_test.sh']:
        shutil.copy2(ROOT/'scripts/viz'/name,args.output/name)
    print(json.dumps(provenance['summary'],indent=2),flush=True)
    print('DONE',args.output/'index.html',flush=True)


if __name__=='__main__':main()
