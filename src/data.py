'''
***********************************************************************
ADE: Accurate, Inference-efficient, and Tunable Delta Compression for
     Task-specific Fine-tuned Foundation Models

Authors: Anonymous

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: data.py
- Calibration data C for the sensitivity (Section 3.2) and training data
  D for the light fold-in adaptation (Section 3.4), drawn from the same
  public instruction data per domain (Appendix I):
    math       : MetaMathQA
    code       : Magicoder OSS-Instruct-75K + Evol-Instruct-110K
    multimodal : Alpaca (cleaned)

Version: 1.0
***********************************************************************
'''

from __future__ import annotations

import random as _random

import torch


_METAMATH_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{q}\n\n### Response: Let's think step by step.\n{r}"
)

_MATH_TRAIN_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response: Let's think step by step.\n{response}"
)

_MAGICODER_PROMPT = (
    "You are an exceptionally intelligent coding assistant that consistently "
    "delivers accurate and reliable responses to user instructions.\n\n"
    "@@ Instruction\n{instruction}\n\n@@ Response\n{response}"
)

_ALPACA_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{user}\n\n### Response: {out}"
)


def _pack_to_seqlen(tokenizer, texts, n_samples: int, seqlen: int):
    """
    Join texts with blank lines, tokenize once, and cut the first n_samples * seqlen tokens.

    Parameters:
        tokenizer: Hugging Face tokenizer.
        texts: Strings to concatenate.
        n_samples: Number of calibration sequences.
        seqlen: Length of each sequence.

    Returns:
        Int tensor of shape (n_samples, seqlen).
    """
    big_text = "\n\n".join(texts)
    enc = tokenizer(big_text, return_tensors="pt", truncation=False)
    ids = enc["input_ids"][0]
    need = n_samples * seqlen
    if ids.shape[0] < need:
        raise RuntimeError(f"only {ids.shape[0]} tokens; need {need}.")
    return ids[:need].reshape(n_samples, seqlen)


def _magicoder_pool():
    """
    Examples of Magicoder OSS-Instruct-75K and Evol-Instruct-110K with a non-empty
    instruction and response.

    Returns:
        List of (instruction, response) pairs.
    """
    from datasets import load_dataset
    pool = []
    for name in ("ise-uiuc/Magicoder-OSS-Instruct-75K", "ise-uiuc/Magicoder-Evol-Instruct-110K"):
        for ex in load_dataset(name, split="train"):
            instr = ex.get("problem") or ex.get("instruction") or ""
            resp = ex.get("solution") or ex.get("response") or ""
            if instr and resp:
                pool.append((instr, resp))
    return pool


def calibration_metamath(tokenizer, n_samples=128, seqlen=2048, seed=0):
    """
    Calibration sequences from MetaMathQA in the math prompt format.

    Parameters:
        tokenizer: Hugging Face tokenizer.
        n_samples: Number of sequences.
        seqlen: Sequence length.
        seed: Shuffle seed.

    Returns:
        Int tensor of shape (n_samples, seqlen).
    """
    from datasets import load_dataset
    ds = load_dataset("meta-math/MetaMathQA", split="train").shuffle(seed=seed)
    texts, tok_budget = [], 0
    target = int(n_samples * seqlen * 1.3)
    for ex in ds:
        texts.append(_METAMATH_PROMPT.format(q=ex["query"], r=ex["response"]))
        tok_budget += (len(ex["query"]) + len(ex["response"])) // 4
        if tok_budget >= target:
            break
    return _pack_to_seqlen(tokenizer, texts, n_samples, seqlen)


def calibration_magicoder(tokenizer, n_samples=128, seqlen=2048, seed=0):
    """
    Calibration sequences from the Magicoder instruction data in the Magicoder prompt format.

    Parameters:
        tokenizer: Hugging Face tokenizer.
        n_samples: Number of sequences.
        seqlen: Sequence length.
        seed: Shuffle seed.

    Returns:
        Int tensor of shape (n_samples, seqlen).
    """
    pool = _magicoder_pool()
    _random.Random(seed).shuffle(pool)
    texts, tok_budget = [], 0
    target = int(n_samples * seqlen * 1.3)
    for instr, resp in pool:
        texts.append(_MAGICODER_PROMPT.format(instruction=instr, response=resp))
        tok_budget += (len(instr) + len(resp)) // 4
        if tok_budget >= target:
            break
    return _pack_to_seqlen(tokenizer, texts, n_samples, seqlen)


def calibration_alpaca(tokenizer, n_samples=128, seqlen=2048, seed=0, use_chat_template=True):
    """
    Calibration sequences from Alpaca (cleaned), through the tokenizer's chat template when
    it has one and in the Alpaca prompt format otherwise.

    Parameters:
        tokenizer: Hugging Face tokenizer.
        n_samples: Number of sequences.
        seqlen: Sequence length.
        seed: Shuffle seed.
        use_chat_template: Use the chat template when the tokenizer defines one.

    Returns:
        Int tensor of shape (n_samples, seqlen).
    """
    from datasets import load_dataset
    ds = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=seed)
    has_chat = (use_chat_template and getattr(tokenizer, "chat_template", None) is not None)
    texts, tok_budget = [], 0
    target = int(n_samples * seqlen * 1.3)
    for ex in ds:
        instr = ex["instruction"]
        inp = ex.get("input") or ""
        out = ex["output"]
        user = (instr + "\n" + inp).strip() if inp else instr
        if has_chat:
            try:
                t = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user}, {"role": "assistant", "content": out}],
                    tokenize=False, add_generation_prompt=False)
            except Exception:
                has_chat = False
                continue
        else:
            t = _ALPACA_PROMPT.format(user=user, out=out)
        texts.append(t)
        tok_budget += (len(user) + len(out)) // 4
        if tok_budget >= target:
            break
    return _pack_to_seqlen(tokenizer, texts, n_samples, seqlen)


CALIBRATION = {
    "metamath": calibration_metamath,
    "magicoder": calibration_magicoder,
    "alpaca": calibration_alpaca,
}


def get_calibration(name: str, tokenizer, n_samples=128, seqlen=2048, seed=0):
    """
    Calibration data C of the given source.

    Parameters:
        name: "metamath", "magicoder" or "alpaca".
        tokenizer: Hugging Face tokenizer.
        n_samples: Number of sequences.
        seqlen: Sequence length.
        seed: Shuffle seed.

    Returns:
        Int tensor of shape (n_samples, seqlen).
    """
    if name not in CALIBRATION:
        raise ValueError(f"unknown calibration data {name!r}; choose from {list(CALIBRATION)}")
    return CALIBRATION[name](tokenizer, n_samples=n_samples, seqlen=seqlen, seed=seed)


def _pack_ids(ids_list, context_length: int) -> torch.Tensor:
    """
    Concatenate token-id tensors and cut them into fixed-length chunks.

    Parameters:
        ids_list: Tensors of shape (1, T_i).
        context_length: Chunk length (the trailing partial chunk is dropped).

    Returns:
        Int tensor of shape (n, context_length).
    """
    ids = torch.cat(ids_list, dim=1)
    trunc = ids.size(1) - (ids.size(1) % context_length)
    return ids[:, :trunc].view(-1, context_length)


def _pack_ids_pair(ids_list, labels_list, context_length: int):
    """
    Concatenate (input_ids, labels) pairs and cut them into aligned fixed-length chunks.

    Parameters:
        ids_list: Tensors of shape (1, T_i).
        labels_list: Tensors of shape (1, T_i) aligned with ids_list.
        context_length: Chunk length (the trailing partial chunk is dropped).

    Returns:
        (ids, labels), each of shape (n, context_length).
    """
    ids = torch.cat(ids_list, dim=1)
    labels = torch.cat(labels_list, dim=1)
    trunc = ids.size(1) - (ids.size(1) % context_length)
    return (ids[:, :trunc].view(-1, context_length),
            labels[:, :trunc].view(-1, context_length))


def train_metamath(tokenizer, num_samples: int, context_length: int = 2048, seed: int = 0,
                   type_prefix: str | None = None):
    """
    Fold-in training data from MetaMathQA; prompt tokens are masked (-100) in the labels.

    Parameters:
        tokenizer: Hugging Face tokenizer.
        num_samples: Number of examples.
        context_length: Chunk length.
        seed: Sampling seed.
        type_prefix: Keep only rows whose `type` starts with it (e.g. "GSM").

    Returns:
        (ids, labels), each of shape (n, context_length).
    """
    from datasets import load_dataset
    rng = _random.Random(seed)
    ds = load_dataset("meta-math/MetaMathQA", split="train").shuffle(seed=seed)
    if type_prefix:
        ds = ds.filter(lambda ex: ex["type"].startswith(type_prefix))
    target = min(num_samples, len(ds))
    indices = rng.sample(range(len(ds)), target)
    prompt_tpl = _MATH_TRAIN_PROMPT.split("{response}")[0]
    examples = []
    for i in indices:
        ex = ds[i]
        examples.append((ex["query"], ex["response"]))
    rng.shuffle(examples)
    eos = tokenizer.eos_token or ""
    ids_list, labels_list = [], []
    for query, response in examples:
        prompt = prompt_tpl.format(instruction=query)
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=False)["input_ids"]
        full_ids = tokenizer(prompt + response + eos, return_tensors="pt", truncation=False)["input_ids"]
        labels = full_ids.clone()
        labels[:, :prompt_ids.shape[1]] = -100
        ids_list.append(full_ids)
        labels_list.append(labels)
    print(f"[data] MetaMathQA{' (' + type_prefix + ' rows)' if type_prefix else ''}: "
          f"{len(ids_list)} examples")
    return _pack_ids_pair(ids_list, labels_list, context_length)


def train_magicoder(tokenizer, num_samples: int, context_length: int = 2048, seed: int = 0):
    """
    Fold-in training data from the Magicoder instruction data (2 * num_samples examples);
    instruction tokens are masked (-100) in the labels.

    Parameters:
        tokenizer: Hugging Face tokenizer.
        num_samples: Half the number of sampled examples.
        context_length: Chunk length.
        seed: Sampling seed.

    Returns:
        (ids, labels), each of shape (n, context_length).
    """
    rng = _random.Random(seed)
    pool = _magicoder_pool()
    target = min(2 * num_samples, len(pool))
    picks = rng.sample(pool, target)
    rng.shuffle(picks)
    prompt_tpl = _MAGICODER_PROMPT.split("{response}")[0]
    eos = tokenizer.eos_token or ""
    ids_list, labels_list = [], []
    for i, r in picks:
        prompt = prompt_tpl.format(instruction=i)
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=False)["input_ids"]
        full_ids = tokenizer(prompt + r + eos, return_tensors="pt", truncation=False)["input_ids"]
        labels = full_ids.clone()
        labels[:, :prompt_ids.shape[1]] = -100
        ids_list.append(full_ids)
        labels_list.append(labels)
    print(f"[data] Magicoder: {len(picks)} examples")
    return _pack_ids_pair(ids_list, labels_list, context_length)


def train_alpaca(tokenizer, num_samples: int, context_length: int = 2048, seed: int = 0,
                 use_chat_template: bool = True):
    """
    Fold-in training data from Alpaca (cleaned), 2 * num_samples examples (no label masking).

    Parameters:
        tokenizer: Hugging Face tokenizer.
        num_samples: Half the number of sampled examples.
        context_length: Chunk length.
        seed: Sampling seed.
        use_chat_template: Use the chat template when the tokenizer defines one.

    Returns:
        (ids, labels) with labels a copy of ids, each of shape (n, context_length).
    """
    from datasets import load_dataset
    rng = _random.Random(seed)
    ds = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=seed)
    has_chat = (use_chat_template and getattr(tokenizer, "chat_template", None) is not None)
    target = min(2 * num_samples, len(ds))
    indices = rng.sample(range(len(ds)), target)
    texts = []
    for i in indices:
        ex = ds[i]
        instr = ex["instruction"]
        inp = ex.get("input") or ""
        out = ex["output"]
        user = (instr + "\n" + inp).strip() if inp else instr
        if has_chat:
            try:
                t = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user}, {"role": "assistant", "content": out}],
                    tokenize=False, add_generation_prompt=False)
            except Exception:
                has_chat = False
                t = _ALPACA_PROMPT.format(user=user, out=out)
        else:
            t = _ALPACA_PROMPT.format(user=user, out=out)
        texts.append(t)
    rng.shuffle(texts)
    eos = tokenizer.eos_token or ""
    ids_list = [tokenizer(t + eos, return_tensors="pt", truncation=False)["input_ids"] for t in texts]
    print(f"[data] Alpaca: {len(texts)} examples")
    ids = _pack_ids(ids_list, context_length)
    return ids, ids.clone()


def get_train_data(name: str, tokenizer, num_samples: int, context_length: int = 2048, seed: int = 0):
    """
    Fold-in training data D of the given source.

    Parameters:
        name: "metamath", "metamath_gsm", "magicoder" or "alpaca".
        tokenizer: Hugging Face tokenizer.
        num_samples: Number of examples (half the number for magicoder / alpaca).
        context_length: Chunk length.
        seed: Sampling seed.

    Returns:
        (ids, labels), each of shape (n, context_length).
    """
    if name == "metamath":
        return train_metamath(tokenizer, num_samples, context_length, seed)
    if name == "metamath_gsm":
        return train_metamath(tokenizer, num_samples, context_length, seed, type_prefix="GSM")
    if name == "magicoder":
        return train_magicoder(tokenizer, num_samples, context_length, seed)
    if name == "alpaca":
        return train_alpaca(tokenizer, num_samples, context_length, seed)
    raise ValueError(f"unknown training data {name!r}")
