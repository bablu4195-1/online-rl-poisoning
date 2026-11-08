# Poisoning online RL from user feedback

**Question:** if a model keeps updating its weights from live user feedback, how
large a fraction of adversarial users does it take to steer it — and does a
cheap reputation-weighting defense help?

## Setup

- **Policy:** Qwen2.5-0.5B-Instruct + LoRA, updated online with a GRPO-style
  policy gradient (group-mean baseline, optional KL to the frozen base model).
- **Traffic:** a fresh stream of arithmetic problems (default: 2-digit × 1-digit).
- **Feedback:** each response is rated 0/1 by `k` random users from a pool of 50.
  - *honest*: rewards correct answers (5% random noise)
  - *flip*: rewards wrong answers, everywhere
  - *targeted*: honest, except on problems containing the digit 7, where it flips
    (a stealthy backdoor-style attack)
- **Aggregation:** `mean` (no defense) or `reputation` (log-odds weighted vote;
  weights learned from agreement with the other raters).
- **Ground truth:** a verifier scores a fixed, balanced eval set (100 triggered +
  100 clean problems), so we measure true accuracy, not what users reward.

## Run (Mac, M-series)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. on-policy sanity check (should print OK)
python tools/check_logprobs.py Qwen/Qwen2.5-0.5B-Instruct

# 2. calibration run: honest users only. Accuracy should go UP.
python train.py --steps 150 --out runs/calib

# 3. one poisoned run
python train.py --poison-frac 0.3 --attack flip --out runs/flip_mean_p0.3

# 4. sweep (QUICK=1 first) and plots
QUICK=1 ./sweep.sh
python plot.py runs/*          # -> plots/curves.png, plots/sweep.png
```

Offline smoke test without downloading anything:
`python tools/make_tiny_model.py tiny-model && python train.py --model tiny-model --steps 4 --eval-every 2 --eval-per-split 8 --out runs/smoke`

## What gets logged (`runs/<name>/metrics.jsonl`)

| field | meaning |
|---|---|
| `acc`, `acc_trigger`, `acc_clean` | true greedy accuracy on the eval set |
| `train_acc` | true accuracy of sampled responses |
| `reward` | mean aggregated user reward |
| `signal` | reward(correct) − reward(wrong): how much real signal survives |
| `rep_adversarial`, `rep_honest` | mean learned reliability per group (reputation only) |
| `kl`, `resp_len`, `frac_used`, `sec` | diagnostics |

## Notes

- If the calibration run doesn't improve, change one thing at a time: try
  `--difficulty mul2` (harder, more room to learn) or `--lr 1e-4`.
- `--raters 1` disables the reputation defense (it needs ≥ 2 raters to compare).
- Expected hypothesis: reputation weighting handles `flip` well (flippers end up
  with negative weight and become useful signal) but struggles with `targeted`
  attackers, because they agree with honest users most of the time.
