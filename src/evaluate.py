'''
***********************************************************************
ADE: Accurate, Inference-efficient, and Tunable Delta Compression for
     Task-specific Fine-tuned Foundation Models

Authors: Anonymous

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: evaluate.py
- Benchmarks of Table 1 (Section 4.1), generated with Hugging Face
  generate (greedy decoding) on the model of serving.py, whose compressed
  delta is applied by ADE's kernels:
    math       : GSM8K accuracy (WizardMath prompt)
    code       : MBPP pass@1 of EvalPlus (Magicoder prompt; base tests of
                 the MBPP+ problems)
    multimodal : TextVQA accuracy (LLaVA-1.5 prompt with OCR tokens)
- evaluate() dispatches on the task of the configuration.

Version: 1.0
***********************************************************************
'''

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from math import isclose
from pathlib import Path
from typing import Union


# ----------------------------------------------------------------------
# GSM8K
# ----------------------------------------------------------------------
def _is_digit(s):
    """
    Check whether a value parses as a number.

    Parameters:
        s: Value to test.

    Returns:
        True if `s` can be cast to float after removing commas.
    """
    try:
        float(str(s).replace(",", ""))
        return True
    except ValueError:
        return False


def _is_number(s):
    """
    Check whether a string represents a number (including unicode numerics).

    Parameters:
        s: Value to test.

    Returns:
        True if `s` parses as a number.
    """
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def _math_equal(prediction: Union[bool, float, str], reference: Union[float, str]) -> bool:
    """
    Numerical equality of a predicted and a reference answer (relative tolerance 1e-4,
    also accepting the reference scaled by 1/100 or 100).

    Parameters:
        prediction: Predicted answer.
        reference: Ground-truth answer.

    Returns:
        True if the prediction matches the reference.
    """
    if _is_digit(prediction) and _is_digit(reference):
        prediction = float(str(prediction).replace(",", ""))
        reference = float(str(reference).replace(",", ""))
        for item in (reference / 100, reference, reference * 100):
            if isclose(item, prediction, rel_tol=1e-4):
                return True
        return False
    return str(prediction).strip() == str(reference).strip()


def _extract_answer_number(completion):
    """
    Extract the final numeric answer after "The answer is: " in a completion.

    Parameters:
        completion: Generated text.

    Returns:
        The rounded numeric answer, or None if none is found.
    """
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if _is_number(denominator) and _is_number(numerator):
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    frac = Fraction(match.group().replace(',', ''))
                    return round(float(frac.numerator / frac.denominator))
                return None
            if float(match.group().replace(',', '')) == float('inf'):
                return None
            return round(float(match.group().replace(',', '')))
    return None


_GSM8K_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
)
_GSM8K_STOP = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT",
               "Instruction:", "Instruction", "Response:", "Response"]
_GSM8K_MAX_TOKENS = 1024


def _gsm8k_items(limit: int = 0):
    """
    GSM8K test prompts and integer answers.

    Parameters:
        limit: Keep only the first `limit` problems (0 = all 1319).

    Returns:
        (prompts, answers).
    """
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    prompts, answers = [], []
    for item in ds:
        prompts.append(_GSM8K_PROMPT.format(instruction=item["question"]))
        answers.append(int(item['answer'].split('#### ')[1].replace(',', '')))
    if limit and limit > 0:
        prompts, answers = prompts[:limit], answers[:limit]
    return prompts, answers


def _gsm8k_score(completions, answers) -> float:
    """
    GSM8K accuracy of a list of completions.

    Parameters:
        completions: Generated texts.
        answers: Integer answers.

    Returns:
        Accuracy in [0, 1].
    """
    result = []
    for completion, answer in zip(completions, answers):
        y_pred = _extract_answer_number(completion)
        result.append(y_pred is not None and (float(y_pred) == float(answer) or _math_equal(y_pred, answer)))
    acc = sum(result) / len(result)
    print(f"[gsm8k] n={len(result)} acc={acc:.4f}")
    return acc


def _cut_at_stop(text: str, stops) -> str:
    """
    Truncate a completion at the first occurrence of any stop string.

    Parameters:
        text: Generated text.
        stops: Stop strings.

    Returns:
        Text before the earliest stop string.
    """
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


def _greedy_generation_config(model):
    """
    Pin the model's generation config to greedy decoding.

    Parameters:
        model: Hugging Face model.
    """
    model.generation_config.do_sample = False
    for attr in ("temperature", "top_p", "top_k"):
        setattr(model.generation_config, attr, None)


def evaluate_gsm8k(model, tokenizer, limit: int = 0, batch_size: int = 32) -> float:
    """
    GSM8K test accuracy (greedy decoding, at most 1024 new tokens, WizardMath stop strings).

    Parameters:
        model: Model (e.g. from serving.build_kernel_model).
        tokenizer: Its tokenizer.
        limit: Evaluate only the first `limit` problems (0 = all 1319).
        batch_size: Generation batch size.

    Returns:
        Accuracy in [0, 1].
    """
    import torch
    prompts, answers = _gsm8k_items(limit)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    _greedy_generation_config(model)
    completions = []
    with torch.no_grad():
        for b in range(0, len(prompts), batch_size):
            enc = tokenizer(prompts[b:b + batch_size], return_tensors="pt", padding=True).to(model.device)
            out = model.generate(**enc, do_sample=False, max_new_tokens=_GSM8K_MAX_TOKENS,
                                 stop_strings=_GSM8K_STOP, tokenizer=tokenizer,
                                 pad_token_id=tokenizer.pad_token_id)
            texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            completions += [_cut_at_stop(t, _GSM8K_STOP) for t in texts]
            print(f"[gsm8k] {len(completions)}/{len(prompts)} generated")
    return _gsm8k_score(completions, answers)


# ----------------------------------------------------------------------
# MBPP (EvalPlus)
# ----------------------------------------------------------------------
_MAGICODER_PROMPT = (
    "You are an exceptionally intelligent coding assistant that "
    "consistently delivers accurate and reliable responses to user "
    "instructions.\n\n"
    "@@ Instruction\n{instruction}\n\n"
    "@@ Response\n{response}"
)


def _map_mbpp_problem(p: dict) -> tuple[str, str, str]:
    """
    Turn an MBPP+ problem into (task_id, instruction, response prefix).

    Parameters:
        p: EvalPlus problem dict with "task_id" and "prompt".

    Returns:
        (task_id, instruction with the required assertion, response prefix "```python").
    """
    prompt = p["prompt"]
    start_index = prompt.index('"""')
    end_index = prompt.rindex('"""')
    body = prompt[start_index + 3:end_index]
    assert_index = body.index("assert")
    nl = body[:assert_index].strip()
    if not nl.endswith("."):
        nl += "."
    assertion = body[assert_index:].strip()
    instruction = (f"{nl} Your code should satisfy the following assertion:\n"
                   f"```python\n{assertion}\n```")
    return str(p["task_id"]), instruction, "```python"


def _truncate_at_fence(text: str) -> str:
    """
    Truncate a completion at its first code fence.

    Parameters:
        text: Generated text.

    Returns:
        Text before the first "```".
    """
    idx = text.find("```")
    return text[:idx] if idx != -1 else text


def evaluate_mbpp(model, tokenizer, output_dir: str, problems_per_batch: int = 24) -> dict:
    """
    MBPP pass@1 on the EvalPlus MBPP+ problems (base tests), with greedy generation of at
    most 512 new tokens.

    Parameters:
        model: Model (serving.build_kernel_model).
        tokenizer: Its tokenizer.
        output_dir: Working directory for the samples and the EvalPlus results.
        problems_per_batch: Generation batch size.

    Returns:
        {"pass@1": float, "n": int}.
    """
    import torch
    from transformers import GenerationConfig
    from evalplus.data import get_mbpp_plus, write_jsonl

    work = Path(output_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    items = [_map_mbpp_problem(p) for p in get_mbpp_plus().values()]

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.eval()
    gen_config = GenerationConfig(max_new_tokens=512, top_p=1.0, eos_token_id=tokenizer.eos_token_id,
                                  pad_token_id=tokenizer.pad_token_id, do_sample=False)
    model.generation_config.do_sample = False
    for attr in ("temperature", "top_p", "top_k"):
        setattr(model.generation_config, attr, None)

    bos_id = tokenizer.bos_token_id
    samples = []
    with torch.no_grad():
        for b in range(0, len(items), problems_per_batch):
            batch = items[b:b + problems_per_batch]
            prompts = [_MAGICODER_PROMPT.format(instruction=instr, response=rp) for (_, instr, rp) in batch]
            enc = tokenizer(prompts, add_special_tokens=False, return_tensors=None, padding=False)
            ids_with_bos = [[bos_id] + ids for ids in enc["input_ids"]]
            max_len = max(len(x) for x in ids_with_bos)
            pad_id = tokenizer.pad_token_id
            input_ids = torch.tensor([[pad_id] * (max_len - len(x)) + x for x in ids_with_bos],
                                     dtype=torch.long).to(model.device)
            attention_mask = (input_ids != pad_id).long()
            out = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                 generation_config=gen_config)
            completions = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
            for (tid, _, _), comp in zip(batch, completions):
                samples.append({"task_id": tid, "completion": _truncate_at_fence(comp)})
            print(f"[mbpp] {len(samples)}/{len(items)} generated")
    del model
    torch.cuda.empty_cache()

    samples_path = work / "samples.jsonl"
    write_jsonl(str(samples_path), samples)
    env = os.environ.copy()
    env.setdefault("HF_ALLOW_CODE_EVAL", "1")
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")

    sanitize = (["evalplus.sanitize"] if shutil.which("evalplus.sanitize", path=env["PATH"])
                else [sys.executable, "-m", "evalplus.sanitize"])
    subprocess.run(sanitize + ["--samples", str(samples_path)], cwd=str(work), env=env, check=True)
    hits = sorted(work.glob(samples_path.stem + "*sanitized*.jsonl"))
    eval_input = hits[0] if hits else samples_path
    eval_path = eval_input.with_name(eval_input.stem + "_eval_results.json")
    if eval_path.exists():
        eval_path.unlink()
    evaluate_cmd = (["evalplus.evaluate"] if shutil.which("evalplus.evaluate", path=env["PATH"])
                    else [sys.executable, "-m", "evalplus.evaluate"])
    subprocess.run(evaluate_cmd + ["--dataset", "mbpp", "--samples", str(eval_input),
                                   "--base_only", "--i_just_wanna_run"],
                   cwd=str(work), env=env, check=True)
    ev = json.loads(eval_path.read_text())["eval"]
    n = len(ev)
    passed = sum(1 for v in ev.values() if v[0]["base_status"] == "pass")
    pass1 = passed / n if n else 0.0
    print(f"[mbpp] n={n} pass@1={pass1:.4f}")
    return {"pass@1": pass1, "n": n}


# ----------------------------------------------------------------------
# TextVQA
# ----------------------------------------------------------------------
_VICUNA_SYS = ("A chat between a curious user and an artificial intelligence assistant. "
               "The assistant gives helpful, detailed, and polite answers to the user's questions.")
_TEXTVQA_PROMPT = (f"{_VICUNA_SYS} USER: <image>\nReference OCR token: {{ocr}}\n{{question}}\n"
                   "Answer the question using a single word or phrase. ASSISTANT:")


def _normalize(s: str) -> str:
    """
    Normalize an answer string (lower case, collapsed whitespace, no trailing punctuation).

    Parameters:
        s: Raw string.

    Returns:
        Normalized string.
    """
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.rstrip(".,!?")


_TEXTVQA_STOP = ["\nUSER:", "\n\n"]
_TEXTVQA_MAX_TOKENS = 64


def _textvqa_items(limit: int = 0):
    """
    TextVQA validation questions with their images, OCR tokens and reference answers.

    Parameters:
        limit: Keep only the first `limit` questions (0 = all 5000).

    Returns:
        List of dicts (question, answers, image, ocr_tokens).
    """
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/textvqa", split="validation")
    items = [{"question": ex["question"], "answers": ex["answers"], "image": ex["image"],
              "ocr_tokens": ex.get("ocr_tokens", [])} for ex in ds]
    if limit and limit > 0:
        items = items[:limit]
    return items


def _textvqa_prompt(it) -> str:
    """
    LLaVA-1.5 prompt of a TextVQA question.

    Parameters:
        it: Item with question and ocr_tokens.

    Returns:
        Prompt string.
    """
    return _TEXTVQA_PROMPT.format(question=it["question"], ocr=", ".join(it.get("ocr_tokens") or []))


def _textvqa_score(items, predictions) -> float:
    """
    TextVQA accuracy: min(#matching reference answers / 3, 1) per question.

    Parameters:
        items: TextVQA items.
        predictions: Predicted answers.

    Returns:
        Accuracy in percent.
    """
    total = 0.0
    for it, pred in zip(items, predictions):
        p = _normalize(pred)
        matches = sum(1 for a in it["answers"] if _normalize(a) == p)
        total += min(matches / 3.0, 1.0)
    acc = 100.0 * total / len(items)
    print(f"[textvqa] n={len(items)} acc={acc:.2f}")
    return acc


def evaluate_textvqa(model, processor, limit: int = 0, batch_size: int = 16) -> float:
    """
    TextVQA validation accuracy (greedy decoding, at most 64 new tokens), scored as
    min(#matching reference answers / 3, 1) per question.

    Parameters:
        model: LLaVA model (e.g. from serving.build_kernel_model).
        processor: Its processor.
        limit: Evaluate only the first `limit` questions (0 = all 5000).
        batch_size: Generation batch size.

    Returns:
        Accuracy in percent.
    """
    import torch
    items = _textvqa_items(limit)
    tok = processor.tokenizer
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    _greedy_generation_config(model)
    predictions = []
    with torch.no_grad():
        for b in range(0, len(items), batch_size):
            batch = items[b:b + batch_size]
            inputs = processor(text=[_textvqa_prompt(it) for it in batch], images=[it["image"] for it in batch],
                               return_tensors="pt", padding=True).to(model.device, torch.float16)
            out = model.generate(**inputs, do_sample=False, max_new_tokens=_TEXTVQA_MAX_TOKENS,
                                 stop_strings=_TEXTVQA_STOP, tokenizer=tok, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            predictions += [_cut_at_stop(t, _TEXTVQA_STOP) for t in texts]
            print(f"[textvqa] {len(predictions)}/{len(items)} generated")
    return _textvqa_score(items, predictions)


def evaluate(task: str, model, processor, output_dir: str, limit: int = 0) -> dict:
    """
    Run the Table 1 benchmark of a task and write summary.json.

    Parameters:
        task: "math", "code" or "multimodal".
        model: Model of serving.build_kernel_model.
        processor: Its tokenizer (math, code) or processor (multimodal).
        output_dir: Directory for the benchmark outputs and summary.json.
        limit: Evaluate only the first `limit` items (GSM8K and TextVQA).

    Returns:
        Dict with the benchmark score.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if task == "math":
        out = {"benchmark": "gsm8k", "accuracy": evaluate_gsm8k(model, processor, limit=limit)}
    elif task == "code":
        res = evaluate_mbpp(model, processor, str(Path(output_dir) / "mbpp"))
        out = {"benchmark": "mbpp_evalplus", "pass@1": res["pass@1"], "n": res["n"]}
    elif task == "multimodal":
        out = {"benchmark": "textvqa", "accuracy": evaluate_textvqa(model, processor, limit=limit)}
    else:
        raise ValueError(f"unknown task {task!r}")
    (Path(output_dir) / "summary.json").write_text(json.dumps(out, indent=2))
    return out
