#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f /usr/local/PPU_SDK/envsetup.sh ]]; then
    source /usr/local/PPU_SDK/envsetup.sh ppu
fi
source /mnt/zhaozc_workspace/envs/drivorpython/bin/activate
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$ROOT"
exec python scripts/viz/visualize_steer_tokens.py \
    --checkpoint "${CHECKPOINT:-$ROOT/exp/ke/steer_stage2_capatch_vitb_8gpu_b16_12ep/11.13_05.07/lightning_logs/version_0/checkpoints/best-epoch=10-step=8866.ckpt}" \
    --test-dir /mnt/workspace/project/LQH/Driv-calude_data/DrivoR-main/outputs/stage2_focus_v2_full_p3/test \
    --output "${OUTPUT:-$ROOT/exp/visualizations/steer_stage2_tokens_test_$(date -u +%Y%m%dT%H%M%SZ)}" \
    "$@"
