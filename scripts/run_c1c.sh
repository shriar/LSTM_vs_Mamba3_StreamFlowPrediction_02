#!/usr/bin/env bash
# C1c: Tier 3 benchmark, resumable queue.
#
# 12 cells = 2 models x 3 depths x 2 seeds at Tier 3 (~1M params), locked
# recipes, 60 epochs, early stop, then exactly one eval_test call per cell.
# Same resume logic as run_c1a.sh:
#   * a cell whose run dir already has master.json skips training;
#   * a partial run dir (no master.json) is deleted and trained from scratch;
#   * a cell whose results/test dir already has test_metrics.csv is skipped.
#
# Usage: bash experiment/scripts/run_c1c.sh
set -uo pipefail
EXP=/mnt/d/Extra/playground-mine/001_Mamba_vs_LSTM_Streamflow_prediction/002_Streamflow_prediction/experiment
cd "$EXP" || exit 1
P=/home/dev/miniconda3/envs/python313/bin/python
mkdir -p logs

CELLS=$("$P" - <<'EOF'
import json
m = json.load(open("recipes/eval_manifest.json"))
for c in m["cells"]:
    if c["id"].startswith("C1c-"):
        print("|".join(str(c[k]) for k in ("id", "recipe", "tier", "depth", "seed", "run_dir")))
EOF
)

FAILED=()
for row in $CELLS; do
  IFS='|' read -r cell recipe tier depth seed run_dir <<< "$row"
  tag="c1c_${cell}"
  echo "=== ${cell} ==="

  if [ -f "${run_dir}/master.json" ]; then
    echo "  train: already complete, skip"
  else
    if [ -d "$run_dir" ]; then
      echo "  train: removing partial run dir"
      rm -rf "$run_dir"
    fi
    echo "  train: ${run_dir}"
    if ! "$P" src/train_val.py --model "$([[ $recipe == *mamba* ]] && echo mamba3 || echo lstm)" \
        --tier "$tier" --depth "$depth" --seed "$seed" --epochs 60 \
        --config "$recipe" > "logs/${tag}_train.log" 2>&1; then
      echo "  TRAIN FAILED (see logs/${tag}_train.log)"
      FAILED+=("${cell}-train")
      continue
    fi
  fi

  recipe_id=$("$P" -c "import json;print(json.load(open('$recipe'))['id'])")
  test_dir="results/test/${recipe_id}_T${tier}_L${depth}_Seed${seed}"
  if [ -f "${test_dir}/test_metrics.csv" ]; then
    echo "  eval: already complete, skip"
    continue
  fi
  echo "  eval: ${test_dir}"
  if ! "$P" src/eval_test.py --recipe "$recipe" --tier "$tier" --depth "$depth" \
      --seed "$seed" > "logs/${tag}_test.log" 2>&1; then
    echo "  EVAL FAILED (see logs/${tag}_test.log)"
    FAILED+=("${cell}-eval")
  else
    grep -h "Test median NSE" "logs/${tag}_test.log" | tail -1
  fi
done

echo "=== C1c queue finished ==="
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED: ${FAILED[*]}"
  exit 1
fi
echo "ALL 12 CELLS COMPLETE"
