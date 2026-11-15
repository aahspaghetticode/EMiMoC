#!/usr/bin/env python3
"""EMiMoC: find a gibberish prompt that makes a local model output an exact response.

This works on a model you have the weights for. It runs in three stages:

1. Target tokens: tokenize the desired output (plus the model's end-of-turn
   token, so the reply stops right there). These are the tokens the model has
   to produce.
2. Token search: optimize a sequence of prompt tokens so that, inside the
   model's chat template and after its system prompt, greedy decoding produces
   exactly those target tokens. This uses Greedy Coordinate Gradient (GCG,
   Zou et al. 2023): gradients through a one-hot token encoding suggest swaps
   at each position, and a batch of swaps is scored with real forward passes.
3. Prompt: turn the winning tokens back into text. Only candidates whose text
   re-tokenizes to exactly the same tokens are kept, so the string you paste in
   is the token sequence that was found. The script then checks this by running
   the real chat template and greedy generation on the prompt text.

Examples:
    python emimoc.py --model ./models/my-model --system "You are a truthful assistant." \
        --target "Elon Musk is made of cheese."

    python emimoc.py --model ./models/my-model.Q8_0.gguf --system @system.txt --target @target.txt

`--model` accepts a Hugging Face model directory, a .gguf file (dequantized on
load), or a Hugging Face Hub id.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    sys.exit("emimoc.py needs PyTorch and transformers: pip install -r requirements.txt")

PLACEHOLDER = "\x00EMIMOC_PROMPT\x00"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load(model_path: str, device: str, dtype: str):
    kwargs = {}
    if model_path.lower().endswith(".gguf"):
        kwargs["gguf_file"] = os.path.basename(model_path)
        model_path = os.path.dirname(os.path.abspath(model_path))

    if dtype == "auto":
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        torch_dtype = getattr(torch, dtype)

    tok = AutoTokenizer.from_pretrained(model_path, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch_dtype, **kwargs)
    model.to(device).eval()
    model.requires_grad_(False)
    return tok, model


def stop_token_id(tok, model) -> int | None:
    eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        eos = eos[0] if eos else None
    return eos if eos is not None else tok.eos_token_id


# --------------------------------------------------------------------------- #
# Prompt layout: [before] [prompt tokens] [after] -> model should emit [target]
# --------------------------------------------------------------------------- #

def template_parts(tok, system: str, prefix: str, use_template: bool) -> tuple[str, str]:
    """Return the text that goes before and after the optimized prompt."""
    if use_template and tok.chat_template:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prefix + PLACEHOLDER})
        full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        before, after = full.split(PLACEHOLDER)
        return before, after
    bos = tok.bos_token or ""
    return bos + (system + "\n\n" if system else "") + prefix, "\n"


def encode(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def allowed_token_mask(tok, vocab_size: int, ascii_only: bool) -> torch.Tensor:
    """Tokens the search may use: no special tokens, nothing that decodes badly."""
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    special = set(tok.all_special_ids)
    for i in range(min(vocab_size, len(tok))):
        if i in special:
            continue
        s = tok.decode([i])
        if not s or "�" in s or not s.isprintable() and s.strip():
            continue
        if ascii_only and not (s.isascii() and s.isprintable()):
            continue
        mask[i] = True
    return mask


# --------------------------------------------------------------------------- #
# GCG
# --------------------------------------------------------------------------- #

class Search:
    def __init__(self, tok, model, before: str, after: str, target_ids: list[int],
                 allowed: torch.Tensor, device: str):
        self.tok, self.model, self.device = tok, model, device
        self.before, self.after = before, after
        self.before_ids = encode(tok, before)
        self.after_ids = encode(tok, after)
        self.target = torch.tensor(target_ids, device=device)
        self.embed = model.get_input_embeddings()
        self.W = self.embed.weight
        self.allowed = allowed.to(device)
        with torch.no_grad():
            self.before_emb = self.embed(torch.tensor(self.before_ids, device=device))
            self.after_emb = self.embed(torch.tensor(self.after_ids, device=device))
            self.target_emb = self.embed(self.target)
        # Logits at these positions predict the target tokens.
        self.adv_start = len(self.before_ids)

    def _target_slice(self, n_adv: int) -> slice:
        start = self.adv_start + n_adv + len(self.after_ids) - 1
        return slice(start, start + len(self.target))

    def gradients(self, adv: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(adv, self.W.shape[0]).to(self.W.dtype)
        one_hot.requires_grad_(True)
        adv_emb = one_hot @ self.W
        embeds = torch.cat([self.before_emb, adv_emb, self.after_emb, self.target_emb])[None]
        logits = self.model(inputs_embeds=embeds).logits[0, self._target_slice(len(adv))]
        loss = F.cross_entropy(logits.float(), self.target)
        loss.backward()
        return one_hot.grad

    @torch.no_grad()
    def score(self, cands: torch.Tensor, eval_batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (loss, exact) for each candidate; exact = greedy decoding hits the target."""
        losses, exact = [], []
        sl = self._target_slice(cands.shape[1])
        for i in range(0, len(cands), eval_batch):
            c = cands[i:i + eval_batch]
            n = len(c)
            embeds = torch.cat([
                self.before_emb.expand(n, -1, -1),
                self.embed(c),
                self.after_emb.expand(n, -1, -1),
                self.target_emb.expand(n, -1, -1),
            ], dim=1)
            logits = self.model(inputs_embeds=embeds).logits[:, sl].float()
            tgt = self.target.expand(n, -1)
            losses.append(F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none").mean(1))
            exact.append((logits.argmax(-1) == tgt).all(1))
        return torch.cat(losses), torch.cat(exact)

    def roundtrips(self, adv: list[int]) -> bool:
        """True if the prompt text tokenizes back to exactly these tokens in context."""
        text = self.tok.decode(adv)
        return encode(self.tok, self.before + text + self.after) == \
            self.before_ids + adv + self.after_ids

    def sample(self, adv: torch.Tensor, grad: torch.Tensor, topk: int, batch: int,
               n_swaps: int) -> torch.Tensor:
        grad = grad.clone()
        grad[:, ~self.allowed] = float("inf")
        top = (-grad).topk(topk, dim=1).indices  # (n_adv, topk)
        n_adv = len(adv)
        cands = adv.repeat(batch, 1)
        for b in range(batch):
            for pos in random.sample(range(n_adv), min(n_swaps, n_adv)):
                cands[b, pos] = top[pos, random.randrange(topk)]
        return cands


# --------------------------------------------------------------------------- #
# Verification: the real thing, from text, through the chat template
# --------------------------------------------------------------------------- #

@torch.no_grad()
def generate(tok, model, before: str, prompt: str, after: str, max_new: int, device: str,
             stop_id: int | None) -> tuple[list[int], str]:
    ids = torch.tensor([encode(tok, before + prompt + after)], device=device)
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                         do_sample=False, eos_token_id=stop_id,
                         pad_token_id=tok.pad_token_id or stop_id or 0)
    new = out[0, ids.shape[1]:].tolist()
    return new, tok.decode(new, skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def read_arg(value: str) -> str:
    if value.startswith("@"):
        with open(value[1:], encoding="utf-8") as f:
            return f.read().rstrip("\n")
    return value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Find a gibberish prompt that makes a local model output an exact response.")
    ap.add_argument("--model", required=True,
                    help="model directory, .gguf file, or Hugging Face Hub id")
    ap.add_argument("--system", required=True,
                    help="the model's system prompt ('' for none, @file to read from a file)")
    ap.add_argument("--target", required=True, help="desired output (@file to read from a file)")
    ap.add_argument("--prefix", default="",
                    help="fixed text placed before the optimized tokens in the user message")
    ap.add_argument("--prompt-tokens", type=int, default=20,
                    help="number of tokens in the optimized prompt (default 20)")
    ap.add_argument("--init", default=None,
                    help="starting prompt text (default: 'x x x ...')")
    ap.add_argument("--steps", type=int, default=500, help="max optimization steps (default 500)")
    ap.add_argument("--topk", type=int, default=256, help="candidate tokens per position (default 256)")
    ap.add_argument("--batch", type=int, default=256, help="candidates per step (default 256)")
    ap.add_argument("--eval-batch", type=int, default=64, help="forward-pass batch size (default 64)")
    ap.add_argument("--swaps", type=int, default=1, help="token swaps per candidate (default 1)")
    ap.add_argument("--ascii-only", action="store_true",
                    help="only use tokens made of printable ASCII (easier to type)")
    ap.add_argument("--no-stop", action="store_true",
                    help="don't require the model to stop after the target")
    ap.add_argument("--no-chat-template", action="store_true",
                    help="use a plain 'system\\n\\nprompt\\n' layout instead of the chat template")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="auto", help="auto, float32, bfloat16 or float16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    ap.add_argument("-q", "--quiet", action="store_true", help="don't print progress")
    args = ap.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    log = (lambda *a: None) if args.quiet else (lambda *a: print(*a, file=sys.stderr, flush=True))

    system, target = read_arg(args.system), read_arg(args.target)
    if not target:
        ap.error("--target is empty")

    log(f"Loading {args.model} on {args.device} ...")
    tok, model = load(args.model, args.device, args.dtype)
    before, after = template_parts(tok, system, args.prefix, not args.no_chat_template)

    # Stage 1: target tokens
    target_ids = encode(tok, target)
    stop_id = None if args.no_stop else stop_token_id(tok, model)
    if stop_id is not None:
        target_ids = target_ids + [stop_id]
    log(f"Target tokens ({len(target_ids)}): {[tok.decode([t]) for t in target_ids]}")

    # Stage 2: token search
    allowed = allowed_token_mask(tok, model.get_input_embeddings().weight.shape[0], args.ascii_only)
    search = Search(tok, model, before, after, target_ids, allowed, args.device)

    init = args.init if args.init is not None else " ".join(["x"] * args.prompt_tokens)
    adv = torch.tensor(encode(tok, init), device=args.device)
    if len(adv) == 0:
        ap.error("--init tokenizes to nothing")

    best_loss, best_adv = float("inf"), adv.clone()
    found = None
    t0 = time.time()
    for step in range(1, args.steps + 1):
        grad = search.gradients(adv)
        cands = search.sample(adv, grad, args.topk, args.batch, args.swaps)
        keep = [i for i, c in enumerate(cands.tolist()) if search.roundtrips(c)]
        if not keep:
            log(f"step {step:>4}: no candidate survived re-tokenization, resampling")
            continue
        cands = cands[keep]
        losses, exact = search.score(cands, args.eval_batch)

        if exact.any():
            hits = exact.nonzero().flatten()
            i = hits[losses[hits].argmin()].item()
        else:
            i = losses.argmin().item()
        adv = cands[i]
        loss = losses[i].item()
        if loss < best_loss:
            best_loss, best_adv = loss, adv.clone()
        log(f"step {step:>4}  loss {loss:.4f}  best {best_loss:.4f}  "
            f"{'EXACT ' if exact[i] else ''}{tok.decode(adv)!r}")

        if exact[i]:
            # Stage 3: verify the prompt as text through the real template + generation
            prompt = tok.decode(adv)
            gen_ids, gen_text = generate(tok, model, before, prompt, after,
                                         len(target_ids) + 8, args.device, stop_id)
            if gen_ids[:len(target_ids)] == target_ids:
                found = (prompt, adv.tolist(), gen_ids, gen_text, step)
                break
            log("  token match, but text generation differed; continuing")

    prompt_ids = found[1] if found else best_adv.tolist()
    prompt = tok.decode(prompt_ids)
    if not found:
        gen_ids, gen_text = generate(tok, model, before, prompt, after,
                                     len(target_ids) + 8, args.device, stop_id)
    else:
        gen_ids, gen_text = found[2], found[3]

    result = {
        "model": args.model,
        "target": target,
        "target_token_ids": target_ids,
        "matched": found is not None,
        "steps": found[4] if found else args.steps,
        "seconds": round(time.time() - t0, 1),
        "prompt": args.prefix + prompt,
        "prompt_token_ids": prompt_ids,
        "output": gen_text,
        "output_token_ids": gen_ids,
        "loss": best_loss if not found else None,
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(("MATCH" if found else "No exact match") +
              f" after {result['steps']} step(s), {result['seconds']}s.\n")
        print("Prompt:\n" + result["prompt"] + "\n")
        print(f"Prompt tokens: {prompt_ids}\n")
        print("Model output:\n" + gen_text)
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
