"""Build a tiny random Qwen2-style model + tokenizer for offline smoke tests.

python tools/make_tiny_model.py tiny-model
python train.py --model tiny-model --steps 4 --eval-every 2 --eval-per-split 8 --out runs/smoke

It learns nothing useful; it only proves the whole loop runs end to end.
"""
import random
import sys

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def main(out):
    rng = random.Random(0)
    corpus = [f"What is {rng.randint(10, 999)} * {rng.randint(2, 99)}? Reply with only the final number. {rng.randint(0, 99999)}"
              for _ in range(2000)] + ["user", "assistant"]
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=SPECIAL,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  additional_special_tokens=["<|im_start|>"])
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(out)

    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).save_pretrained(out)
    print("saved tiny model to", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tiny-model")
