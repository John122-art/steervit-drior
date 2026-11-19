"""Verify the completed expansion, update matched-sample conclusions, preserve old output."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import numpy as np

root = Path('/mnt/zhaozc_workspace/project/LQH/Drivor-orial/DrivoR-main')
base = root/'exp/visualizations'
current = base/'steer_stage2_tokens_test_20261119'
work = base/'steer_stage2_tokens_test_20261119_expanded30_work'
archive = base/'steer_stage2_tokens_test_20261119_previous_3scenes'
run = json.loads((work/'run.json').read_text())
validation = json.loads((work/'validation.json').read_text())
assert run['complete'] and validation['passed']
assert run['scene_count']==30 and run['views']==120 and run['condition_forwards']==360
assert run['model_keys_loaded']==509 and not run['missing_keys'] and not run['unexpected_keys']
assert run['pca'] is False and run['conditions']==['correct','empty','unrelated']
assert not archive.exists(), archive
rows = json.loads((work/'selected_samples.json').read_text())
assert len(rows)==len({r['log_name'] for r in rows})==30
prompts = json.loads((work/'unrelated_prompts.json').read_text())
new = json.loads((work/'metrics.json').read_text())
old = json.loads((current/'metrics.json').read_text())
assert len(new)==360
for r in new:
    if r['mode']=='unrelated':
        assert r['text']==prompts[r['scene_token']][r['camera']]
original = {r['scene_token'] for r in old}
assert len(original)==3
same = [r for r in new if r['scene_token'] in original]
def means(records, mode):
    values = [r['target_mass'] for r in records if r['mode']==mode and r['grounding_valid']]
    return {'valid_views':len(values),'mean_target_mass':float(np.mean(values))}
matched = dict(previous={m:means(old,m) for m in ['correct','empty','shuffled']},
               updated={m:means(same,m) for m in ['correct','empty','unrelated']},
               scope='Same original 3 scenes and images. Wrong input intentionally changed from other driving captions to absurd unrelated descriptions.')
target_checks=[]
empty_checks=[]
for row in rows:
    for cam in row['camera_order']:
        rel=Path(row['scene_token'])/cam
        with np.load(work/rel/'features.npz',allow_pickle=False) as a:
            if row['scene_token'] in original:
                with np.load(current/rel/'features.npz',allow_pickle=False) as b:
                    assert np.array_equal(a['target'],b['target'])
                target_checks.append(str(rel))
            if not row['scene_text'][cam]['text'].strip():
                assert np.array_equal(a['ground_probabilities'][0],a['ground_probabilities'][1])
                empty_checks.append(str(rel))
matched['identical_reference_masks']=len(target_checks)
(work/'comparison_to_previous.json').write_text(json.dumps(matched,indent=2)+'\n')
pdf_pages={}
for p in [work/'token_comparison.pdf']+[work/r['scene_token']/'scene_comparison.pdf' for r in rows]:
    count=len(re.findall(rb'/Type\s*/Page\b',p.read_bytes()))
    expected=120 if p.name=='token_comparison.pdf' else 8
    assert count==expected,(p,count)
    pdf_pages[str(p.relative_to(work))]=count
assert len(list(work.glob('*/*/comparison.png')))==120
assert len(list(work.glob('*/*/scene_tokens_*.png')))==360
validation.update(actual_pdf_page_counts=pdf_pages,unrelated_prompt_inputs_match_manifest=True,
                  original_reference_masks_unchanged=len(target_checks),
                  empty_original_equals_empty_control_views=len(empty_checks),
                  completed_scene_count=30,completed_condition_forwards=360)
(work/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
extra='\n## 原来 3 个场景的同样本复核\n\n'
extra+='保留原来 12 张图片，使用相同 checkpoint；12 张参考 mask 逐项一致。以下统计使用其中 11 张有效图片：\n\n'
extra+='| 条件 | 上一版 | 扩展版中的原 3 场景 |\n|---|---:|---:|\n'
for label,a,b in [('原始语言','correct','correct'),('空语言','empty','empty'),('错误语言（类型不同）','shuffled','unrelated')]:
    extra+=f"| {label} | {matched['previous'][a]['mean_target_mass']:.2%} | {matched['updated'][b]['mean_target_mass']:.2%} |\n"
extra+='\n原始/空语言结果基本保持一致；错误文字从其他驾驶场景描述换成离谱无关描述后，原目标响应降低。这比混合样本直接比较两个版本更能说明错误输入类型的影响，但仍不等于逐对象语义匹配能力的证明。\n'
extra+=f'\n上一版 3 场景完整产物已保存在 [previous_3scenes](../{archive.name}/index.html)，其中旧 PCA 图只在归档中保留。当前图集没有 PCA。\n'
extra+=f'\n复现可运行工程脚本（生成新的输出目录）：\n\n```bash\nbash {root}/scripts/viz/run_steer_tokens_test.sh --scene-count 30 --seed 2\n```\n'
(work/'README.md').write_text((work/'README.md').read_text()+extra)
page=(work/'index.html').read_text()
page=page.replace('<nav>',f'<p><a href="comparison_to_previous.json">原 3 场景同样本复核</a> · <a href="../{archive.name}/index.html">上一版归档</a></p><nav>',1)
(work/'index.html').write_text(page)
shutil.copy2('/tmp/steer_stage2_tokens_expanded30_20261119.log',work/'inference.log')
shutil.copy2(__file__,work/'finalize_report.py')
run.update(published_output=str(current),previous_version_directory=str(archive),
           source_sha256=hashlib.sha256((root/'scripts/viz/visualize_steer_tokens.py').read_bytes()).hexdigest(),
           validation=validation,matched_original_three=matched)
(work/'run.json').write_text(json.dumps(run,indent=2,ensure_ascii=False)+'\n')
current.rename(archive)
try:
    work.rename(current)
except BaseException:
    archive.rename(current)
    raise
missing=[x for x in re.findall(r'(?:href|src)="([^"]+)"',(current/'index.html').read_text())
         if not x.startswith('#') and not (current/x).is_file()]
assert not missing,missing
print(json.dumps({'published':str(current),'archive':str(archive),'summary':run['summary'],
    'validation':{k:v for k,v in validation.items() if k!='actual_pdf_page_counts'},'matched_original_three':matched},indent=2))
