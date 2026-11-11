#!/usr/bin/env bash
# C11: mechanism extraction pass, resumable.
#
# Six cells = 2 models x 3 seeds at the Tier-1 headline depths (LSTM 1L,
# Mamba-3 3L).  No training: each cell replays an existing checkpoint on the
# validation slice via src/mechanism_extract.py.  A cell whose output folder
# already has mechanism_meta.json is skipped, so the queue can be re-run.
#
# Usage: bash experiment/scripts/run_c11.sh
set -uo pipefail
EXP=/mnt/d/Extra/playground-mine/001_Mamba_vs_LSTM_Streamflow_prediction/002_Streamflow_prediction/experiment
cd "$EXP" || exit 1
P=/home/dev/miniconda3/envs/python313/bin/python
mkdir -p logs

CELLS=$("$P" - <<'EOF'
import json
keep = {
    ("recipes/lstm.json", 1, 1, 1),
    ("recipes/lstm.json", 1, 1, 2),
    ("recipes/lstm.json", 1, 1, 3),
    ("recipes/mamba3.json", 1, 3, 1),
    ("recipes/mamba3.json", 1, 3, 2),
    ("recipes/mamba3.json", 1, 3, 3),
}
m = json.load(open("recipes/eval_manifest.json"))
for c in m["cells"]:
    key = (c["recipe"], int(c["tier"]), int(c["depth"]), int(c["seed"]))
    if key in keep:
        print("|".join(str(c[k]) for k in ("id", "recipe", "tier", "depth", "seed")))
EOF
)

if [ -z "$CELLS" ]; then
  echo "No C11 cells found in the manifest"
  exit 1
fi

FAILED=()
for row in $CELLS; do
  IFS='|' read -r cell recipe tier depth seed <<< "$row"
  tag="c11_${cell}"
  echo "=== ${cell} ==="
  recipe_id=$("$P" -c "import json,sys;print(json.load(open(sys.argv[1]))['id'])" "$recipe")
  out="results/mechanism/${recipe_id}_T${tier}_L${depth}_Seed${seed}"

  if [ -f "${out}/mechanism_meta.json" ]; then
    echo "  mechanism: already complete, skip"
    continue
  fi
  echo "  mechanism: ${out}"
  "$P" src/mechanism_extract.py --recipe "$recipe" --tier "$tier" \
      --depth "$depth" --seed "$seed" > "logs/${tag}.log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  FAILED (see logs/${tag}.log)"
    FAILED+=("$cell")
    continue
  fi
  tail -4 "logs/${tag}.log" | sed 's/^/  /'
done

echo "=== C11 queue finished ==="
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED: ${FAILED[*]}"
  exit 1
fi
echo "ALL C11 CELLS COMPLETE"
