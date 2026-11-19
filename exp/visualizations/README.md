# SteerViT / DrivoR visualization results

- [30-scene report](steer_stage2_tokens_test_20261119/README.md): 120 camera images, original / empty / absurd unrelated language, no PCA.
- [30-scene HTML gallery](steer_stage2_tokens_test_20261119/index.html): download the directory and open locally to browse.
- [120-page comparison PDF](steer_stage2_tokens_test_20261119/token_comparison.pdf).
- [Previous 3-scene report](steer_stage2_tokens_test_20261119_previous_3scenes/README.md): archived results, including the earlier PCA plots.
- [Visualization script](../../scripts/viz/visualize_steer_tokens.py) and [launcher](../../scripts/viz/run_steer_tokens_test.sh).

PDFs and raw NumPy feature archives (`*.npz`) use Git LFS. Install Git LFS, clone this repository, then run `git lfs pull` to retrieve them. PNG/JPEG images, reports, and scripts are stored directly in Git.

The reports contain the original local dataset and checkpoint paths for reproducibility. Those paths must be adapted to your environment before running the launcher. Model checkpoints and private keys are not included. The recorded statistics describe probability mass inside automatically generated reference masks, not planning performance or dataset-wide localization accuracy.

## Latest color-word controls (2026-11-19)

[White/red and white/black four-camera comparisons](steer_stage2_tokens_test_20261119/COLOR_PROMPT_PAIRS.md), including PNGs, 7-page PDFs, raw features, scripts, and measured distribution changes. Neither example shows a clear color-conditioned target switch; no downstream planning evaluation was run.
