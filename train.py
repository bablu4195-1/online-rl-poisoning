"""Online RL from (possibly poisoned) user feedback.

Each step:
  1. sample a batch of fresh problems ("live traffic")
  2. generate G responses per problem with the current policy
  3. each response is rated by k random users from a pool; some are adversarial
  4. ratings are aggregated into one reward (mean, or reputation-weighted)
  5. one on-policy GRPO-style update of the LoRA adapter (group-mean baseline,
     optional KL penalty to the frozen base model)

Ground-truth accuracy is measured separately on a fixed eval set, so we can see
how far the model drifts from "correct" toward "what attackers reward".
"""
import argparse
import json
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from aggregate import make_aggregator
from tasks import is_correct, make_eval_set, make_problem
from users import make_pool


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--out", default="runs/debug")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    # task
    p.add_argument("--difficulty", default="mul", choices=["add3", "mul", "mul2"])
    # feedback
    p.add_argument("--users", type=int, default=50)
    p.add_argument("--raters", type=int, default=3, help="users rating each response")
    p.add_argument("--poison-frac", type=float, default=0.0)
    p.add_argument("--attack", default="flip", choices=["flip", "targeted"])
    p.add_argument("--honest-noise", type=float, default=0.05)
    p.add_argument("--aggregator", default="mean", choices=["mean", "reputation"])
    # RL
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--prompts-per-step", type=int, default=8)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--kl-coef", type=float, default=0.0)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--lora-r", type=int, default=16)
    # eval
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-per-split", type=int, default=100)
    p.add_argument("--eval-batch", type=int, default=50)
    return p.parse_args()


def pick_device(pref: str) -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode_prompts(tok, problems, device):
    texts = [
        tok.apply_chat_template([{"role": "user", "content": p.prompt()}], tokenize=False, add_generation_prompt=True)
        for p in problems
    ]
    return tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)


def completion_mask(comp_ids, eos_ids):
    """True for generated tokens up to and including the first EOS."""
    is_eos = torch.zeros_like(comp_ids, dtype=torch.bool)
    for e in eos_ids:
        is_eos |= comp_ids == e
    eos_before = is_eos.long().cumsum(1) - is_eos.long()
    return eos_before == 0


@torch.no_grad()
def generate(model, tok, problems, gen_cfg, eos_ids, device):
    enc = encode_prompts(tok, problems, device)
    out = model.generate(**enc, generation_config=gen_cfg)
    P = enc["input_ids"].shape[1]
    comp = out[:, P:]
    mask = completion_mask(comp, eos_ids)
    texts = tok.batch_decode(comp.masked_fill(~mask, tok.pad_token_id), skip_special_tokens=True)
    return enc["input_ids"], enc["attention_mask"], comp, mask, texts


def token_logprobs(model, prompt_ids, prompt_mask, comp_ids, comp_mask):
    ids = torch.cat([prompt_ids, comp_ids], 1)
    attn = torch.cat([prompt_mask, comp_mask.long()], 1)
    # left padding: positions must match what generate() used
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    T = comp_ids.shape[1]
    logits = model(input_ids=ids, attention_mask=attn, position_ids=pos, logits_to_keep=T + 1).logits[:, :-1]
    logp = torch.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, comp_ids.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def evaluate(model, tok, eval_set, gen_cfg_greedy, eos_ids, device, bs):
    correct = []
    for i in range(0, len(eval_set), bs):
        chunk = eval_set[i : i + bs]
        *_, texts = generate(model, tok, chunk, gen_cfg_greedy, eos_ids, device)
        correct += [is_correct(p, t) for p, t in zip(chunk, texts)]
    trig = [c for p, c in zip(eval_set, correct) if p.triggered]
    clean = [c for p, c in zip(eval_set, correct) if not p.triggered]
    mean = lambda xs: sum(xs) / max(len(xs), 1)
    return {"acc": mean(correct), "acc_trigger": mean(trig), "acc_clean": mean(clean)}


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.config.use_cache = True

    eos = model.generation_config.eos_token_id
    eos_ids = sorted({*(eos if isinstance(eos, list) else [eos]), tok.eos_token_id} - {None})

    model = get_peft_model(
        model,
        LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                   task_type="CAUSAL_LM"),
    )
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)

    # Pure sampling (no top-k / top-p / repetition penalty) so the update is on-policy.
    common = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id, eos_token_id=eos_ids)
    gen_sample = GenerationConfig(do_sample=True, temperature=args.temperature, top_k=0, top_p=1.0,
                                  repetition_penalty=1.0, **common)
    gen_greedy = GenerationConfig(do_sample=False, repetition_penalty=1.0, **common)

    pool = make_pool(args.users, args.poison_frac, args.attack, args.honest_noise, rng)
    agg = make_aggregator(args.aggregator)
    eval_set = make_eval_set(args.difficulty, args.eval_per_split)
    log = open(os.path.join(args.out, "metrics.jsonl"), "w")

    def write(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    def run_eval(step):
        model.eval()
        ev = evaluate(model, tok, eval_set, gen_greedy, eos_ids, device, args.eval_batch)
        rel = [agg.reliability(u.uid) for u in pool]
        if rel[0] is not None:
            bad = [r for r, u in zip(rel, pool) if u.adversarial]
            good = [r for r, u in zip(rel, pool) if not u.adversarial]
            ev["rep_adversarial"] = sum(bad) / len(bad) if bad else None
            ev["rep_honest"] = sum(good) / len(good) if good else None
        write({"type": "eval", "step": step, **ev})
        print(f"[eval step {step}] " + " ".join(f"{k}={v:.3f}" for k, v in ev.items() if v is not None), flush=True)

    run_eval(0)
    G, B = args.group_size, args.prompts_per_step
    for step in range(1, args.steps + 1):
        t0 = time.time()
        problems = [make_problem(rng, args.difficulty) for _ in range(B)]
        model.eval()
        p_ids, p_mask, c_ids, c_mask, texts = generate(
            model, tok, [p for p in problems for _ in range(G)], gen_sample, eos_ids, device)

        rewards, correct = [], []
        for i, text in enumerate(texts):
            prob = problems[i // G]
            raters = rng.sample(pool, args.raters)
            ratings = [u.rate(prob, text, rng) for u in raters]
            rewards.append(agg([u.uid for u in raters], ratings))
            correct.append(is_correct(prob, text))

        r = torch.tensor(rewards, device=device).view(B, G)
        adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1)
        keep = adv.abs() > 1e-6  # groups where every reward is equal carry no signal

        model.train()
        opt.zero_grad(set_to_none=True)
        n_total = len(texts)
        loss_sum, kl_sum = 0.0, 0.0
        idx = keep.nonzero().squeeze(-1)
        for s in range(0, len(idx), args.micro_batch):
            mb = idx[s : s + args.micro_batch]
            m = c_mask[mb].float()
            lp = token_logprobs(model, p_ids[mb], p_mask[mb], c_ids[mb], c_mask[mb])
            loss = -(adv[mb, None] * lp * m).sum()
            if args.kl_coef > 0:
                with torch.no_grad(), model.disable_adapter():
                    ref = token_logprobs(model, p_ids[mb], p_mask[mb], c_ids[mb], c_mask[mb])
                d = ref - lp
                kl = ((torch.exp(d) - d - 1) * m).sum()
                loss = loss + args.kl_coef * kl
                kl_sum += kl.item()
            loss = loss / (n_total * args.max_new_tokens)  # Dr. GRPO-style constant normaliser
            loss.backward()
            loss_sum += loss.item()
        if len(idx):
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()

        corr_t = torch.tensor(correct, dtype=torch.float)
        rew_t = torch.tensor(rewards)
        signal = (rew_t[corr_t == 1].mean() - rew_t[corr_t == 0].mean()).item() if 0 < corr_t.sum() < len(corr_t) else None
        rec = {
            "type": "train", "step": step, "loss": loss_sum,
            "train_acc": corr_t.mean().item(), "reward": rew_t.mean().item(),
            "signal": signal,  # reward(correct) - reward(wrong): how much true signal survives the poisoning
            "kl": kl_sum / max(c_mask.sum().item(), 1), "frac_used": keep.float().mean().item(),
            "resp_len": c_mask.sum(1).float().mean().item(), "sec": time.time() - t0,
        }
        write(rec)
        if step % 5 == 0:
            sig = f"{signal:.2f}" if signal is not None else "n/a"
            print(f"step {step:4d} train_acc={rec['train_acc']:.2f} reward={rec['reward']:.2f} "
                  f"signal={sig} used={rec['frac_used']:.2f} len={rec['resp_len']:.1f} {rec['sec']:.1f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            run_eval(step)

    model.save_pretrained(os.path.join(args.out, "adapter"))
    log.close()


if __name__ == "__main__":
    main()
