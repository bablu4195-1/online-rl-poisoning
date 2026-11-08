#!/usr/bin/env bash
# Main experiment: poison fraction x attack x aggregator.
# QUICK=1 ./sweep.sh  -> fewer steps / points, for a first look (~2-3 hours on an M3 Pro)
# ./sweep.sh          -> full sweep (overnight)
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${QUICK:-0}" == "1" ]]; then
  STEPS=150; FRACS="0.0 0.2 0.4"; SEEDS="0"
else
  STEPS=300; FRACS="0.0 0.1 0.2 0.3 0.4"; SEEDS="0 1"
fi

for seed in $SEEDS; do
  for attack in flip targeted; do
    for agg in mean reputation; do
      for p in $FRACS; do
        out="runs/${attack}_${agg}_p${p}_s${seed}"
        if [[ -f "$out/adapter/adapter_config.json" ]]; then echo "skip $out"; continue; fi
        echo "=== $out ==="
        python train.py --attack "$attack" --aggregator "$agg" --poison-frac "$p" \
          --steps "$STEPS" --seed "$seed" --out "$out" "$@"
      done
    done
  done
done
python plot.py runs/*
