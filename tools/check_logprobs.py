"""On-policy sanity check: recomputed logprobs must match sampling-time logprobs.

python tools/check_logprobs.py Qwen/Qwen2.5-0.5B-Instruct
"""
# Check: logprobs recomputed by token_logprobs() == logprobs at sampling time (left padding, EOS masking)
import sys, torch; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from train import encode_prompts, completion_mask, token_logprobs
from tasks import make_problem
import random
tok = AutoTokenizer.from_pretrained(sys.argv[1]); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(sys.argv[1], dtype=torch.float32).eval()
rng = random.Random(0)
probs = [make_problem(rng, "mul2" if i % 2 else "mul") for i in range(6)]  # different prompt lengths -> padding
enc = encode_prompts(tok, probs, "cpu")
eos = [tok.eos_token_id]
cfg = GenerationConfig(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0, max_new_tokens=12,
                       pad_token_id=tok.pad_token_id, eos_token_id=eos, output_scores=True, return_dict_in_generate=True)
torch.manual_seed(0)
out = m.generate(**enc, generation_config=cfg)
P = enc["input_ids"].shape[1]; comp = out.sequences[:, P:]
mask = completion_mask(comp, eos)
gen_lp = torch.stack([torch.log_softmax(s.float(), -1) for s in out.scores], 1).gather(-1, comp.unsqueeze(-1)).squeeze(-1)
with torch.no_grad():
    lp = token_logprobs(m, enc["input_ids"], enc["attention_mask"], comp, mask)
diff = ((gen_lp - lp).abs() * mask).max().item()
print("prompt pad counts:", (enc["attention_mask"] == 0).sum(1).tolist(), "| eos hit:", (comp == eos[0]).any(1).tolist())
print("max |logprob diff| over valid tokens:", diff)
assert diff < 1e-3
print("OK")
