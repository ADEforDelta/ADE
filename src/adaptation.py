'''
***********************************************************************
ADE: Accurate, Inference-efficient, and Tunable Delta Compression for
     Task-specific Fine-tuned Foundation Models

Authors: Anonymous

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: adaptation.py
- I3. Light fold-in adaptation on the compressed delta (Section 3.4,
  Appendices G and H).
- FoldInLinear      : W_b + diag(gamma) U^ V^^T diag(rho) (eq. 9) with the
                      packed factors frozen and gamma (R^{d_o}, rows of U^)
                      and rho (R^{d_i}, columns of V^^T) trainable.
- train_fold_in     : L = lambda_LM L_LM + lambda_KD L_KD + lambda_RE L_RE
                      (eq. 20) against the aligned model as the teacher.
- export_folded     : folds gamma into the row scales u and rho into the
                      column scales v (eq. 10); codes and zero-points are
                      unchanged and gamma / rho are discarded.

Version: 1.0
***********************************************************************
'''

from __future__ import annotations

import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR as WarmupLR
from tqdm import tqdm


LINEAR_MODULE_NAMES = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def same_device(a, b) -> bool:
    """
    True when two device specs name the same physical device ("cuda" == "cuda:0").

    Parameters:
        a, b: Device strings or torch.device objects.

    Returns:
        Whether tensors placed on `a` and `b` share one device.
    """
    da, db = torch.device(a), torch.device(b)
    if da.type != db.type:
        return False
    if da.type != "cuda":
        return True
    cur = torch.cuda.current_device() if torch.cuda.is_available() else 0
    return ((da.index if da.index is not None else cur)
            == (db.index if db.index is not None else cur))


def free_memory():
    """
    Run Python GC and (if available) empty the CUDA caching allocator.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _teacher_linear(x, w, pdt):
    """
    h_a(x) = W_a x for L_RE, computed on the device that holds the teacher weight.

    Parameters:
        x: Module input (..., d_i) on the student's device.
        w: Teacher weight W_a (d_o, d_i).
        pdt: Compute dtype.

    Returns:
        W_a x on x's device, dtype pdt.
    """
    if w.device == x.device:
        return F.linear(x.to(pdt), w.to(pdt))
    x_t = x.to(device=w.device, dtype=pdt, non_blocking=True)
    return F.linear(x_t, w.to(pdt)).to(x.device, non_blocking=True)


class _ReconMSE(torch.autograd.Function):
    """
    Per-module term of L_RE, mean((h_ad(x) - h_a(x))^2), with W_a x recomputed in backward.
    """

    @staticmethod
    def forward(ctx, s_full, x, w, pdt):
        """
        Parameters:
            s_full: Output of the adapted module, h_ad(x).
            x: Module input.
            w: Teacher weight W_a.
            pdt: Compute dtype.

        Returns:
            Scalar mean squared error.
        """
        with torch.no_grad():
            t_full = _teacher_linear(x, w, pdt)
            loss = (s_full.float() - t_full.float()).pow(2).mean()
        ctx.save_for_backward(s_full, x, w)
        ctx.pdt = pdt
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        """
        Parameters:
            grad_out: Gradient of the loss.

        Returns:
            Gradient w.r.t. s_full (the other inputs get None).
        """
        s_full, x, w = ctx.saved_tensors
        with torch.no_grad():
            t_full = _teacher_linear(x, w, ctx.pdt)
            grad_s = (s_full.float() - t_full.float()) * (2.0 * grad_out / s_full.numel())
        return grad_s.to(s_full.dtype), None, None, None


class FoldInLinear(nn.Module):
    """
    Linear module W_b + diag(gamma) U^ V^^T diag(rho) of eq. (9); only gamma and rho
    are trainable (initialized to ones).
    """

    def __init__(self, W_b: torch.Tensor, packed: dict, group_size: int,
                 param_dtype=torch.bfloat16):
        """
        Parameters:
            W_b: Base weight (d_o, d_i).
            packed: Dict with U_codes, U_scale, U_zero, Vt_codes, Vt_scale, Vt_zero.
            group_size: Group size G along the rank axis.
            param_dtype: Compute dtype of the module.
        """
        super().__init__()
        self.param_dtype = param_dtype
        self.group_size = group_size
        self.register_buffer("base", W_b.detach().clone().to(param_dtype))
        rank = packed["U_codes"].shape[1]
        if rank % group_size != 0:
            raise ValueError(f"FoldInLinear: rank={rank} not a multiple of group_size={group_size}")
        self._n_groups = rank // group_size
        d_o, d_i = self.base.shape
        # rho before gamma (parameter order)
        self.rho = nn.Parameter(torch.ones(d_i, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.ones(d_o, dtype=torch.float32))
        self.register_buffer("U_codes", packed["U_codes"].to(torch.uint8).clone())
        self.register_buffer("U_scale", packed["U_scale"].to(param_dtype).clone())
        self.register_buffer("U_zero", packed["U_zero"].to(param_dtype).clone())
        self.register_buffer("Vt_codes", packed["Vt_codes"].to(torch.uint8).clone())
        self.register_buffer("Vt_scale", packed["Vt_scale"].to(param_dtype).clone())
        self.register_buffer("Vt_zero", packed["Vt_zero"].to(param_dtype).clone())
        self._recon_teacher_w = None

    def _dequantize_U(self):
        """
        Dequantized U^ (eq. 6) in the compute dtype.

        Returns:
            Tensor of shape (d_o, r).
        """
        gs = self.group_size
        parts = []
        for g in range(self._n_groups):
            cs = slice(g * gs, (g + 1) * gs)
            q = self.U_codes[:, cs].to(self.param_dtype)
            s = self.U_scale[:, g].to(self.param_dtype)
            z = self.U_zero[:, g].to(self.param_dtype)
            parts.append((q - z.unsqueeze(1)) * s.unsqueeze(1))
        return torch.cat(parts, dim=1)

    def _dequantize_Vt(self):
        """
        Dequantized V^^T (eq. 6) in the compute dtype.

        Returns:
            Tensor of shape (r, d_i).
        """
        gs = self.group_size
        parts = []
        for g in range(self._n_groups):
            rs = slice(g * gs, (g + 1) * gs)
            q = self.Vt_codes[rs, :].to(self.param_dtype)
            s = self.Vt_scale[g, :].to(self.param_dtype)
            z = self.Vt_zero[g, :].to(self.param_dtype)
            parts.append((q - z.unsqueeze(0)) * s.unsqueeze(0))
        return torch.cat(parts, dim=0)

    def _delta_path(self, x):
        """
        Factored application diag(gamma) U^ (V^^T diag(rho) x) of the adapted delta.

        Parameters:
            x: Input activations (..., d_i).

        Returns:
            Delta output (..., d_o).
        """
        pdt = self.param_dtype
        V_s = (self._dequantize_Vt().float() * self.rho.unsqueeze(0)).to(pdt)
        U_s = (self._dequantize_U().float() * self.gamma.unsqueeze(1)).to(pdt)
        return torch.matmul(torch.matmul(x, V_s.T), U_s.T)

    def recon_loss(self, x, output):
        """
        This module's term of L_RE (eq. 20): the error between its output h_ad(x) and the
        aligned module's output h_a(x) = W_a x.

        Parameters:
            x: Module input.
            output: The output this module's forward() produced for x.

        Returns:
            None when no teacher weight is attached, else a scalar tensor.
        """
        w = self._recon_teacher_w
        if w is None:
            return None
        return _ReconMSE.apply(output, x, w, self.param_dtype)

    def forward(self, x):
        """
        Parameters:
            x: Input activations (..., d_i).

        Returns:
            x W_b^T + diag(gamma) U^ V^^T diag(rho) x.
        """
        x = x.to(self.param_dtype)
        return F.linear(x, self.base) + self._delta_path(x)

    @torch.no_grad()
    def export_folded(self) -> dict:
        """
        Fold the tuning vectors into the per-group scales (eq. 10): u <- gamma u, v <- rho v.

        Returns:
            Dict with CPU tensors U_codes, U_scale, U_zero, Vt_codes, Vt_scale, Vt_zero.
        """
        gamma = self.gamma.detach().float().cpu()
        rho = self.rho.detach().float().cpu()
        U_scale = (self.U_scale.detach().float().cpu() * gamma.unsqueeze(1)).to(self.U_scale.dtype)
        Vt_scale = (self.Vt_scale.detach().float().cpu() * rho.unsqueeze(0)).to(self.Vt_scale.dtype)
        return {
            "U_codes": self.U_codes.detach().cpu(),
            "U_scale": U_scale,
            "U_zero": self.U_zero.detach().cpu(),
            "Vt_codes": self.Vt_codes.detach().cpu(),
            "Vt_scale": Vt_scale,
            "Vt_zero": self.Vt_zero.detach().cpu(),
        }


def attach_fold_in(model, per_module_packed: dict, group_size: int,
                   param_dtype=torch.bfloat16) -> int:
    """
    Replace every compressed linear module of `model` (a copy of the base model) by a
    FoldInLinear holding its base weight and its packed delta.

    Parameters:
        model: HF causal LM with the base weights.
        per_module_packed: Mapping module prefix ("model.layers.{i}.{name}") -> packed dict.
        group_size: Group size G.
        param_dtype: Compute dtype.

    Returns:
        Number of modules replaced.
    """
    dec = model.model
    n = 0
    for li in range(len(dec.layers)):
        for sn in LINEAR_MODULE_NAMES:
            prefix = f"model.layers.{li}.{sn}"
            if prefix not in per_module_packed:
                continue
            parts = sn.split(".")
            parent = dec.layers[li].get_submodule(".".join(parts[:-1]))
            linear = getattr(parent, parts[-1])
            setattr(parent, parts[-1], FoldInLinear(linear.weight.data, per_module_packed[prefix],
                                                    group_size, param_dtype))
            n += 1
    return n


@torch.no_grad()
def export_folded(model) -> dict:
    """
    Export every FoldInLinear of `model` with gamma / rho folded into the scales (eq. 10).

    Parameters:
        model: Adapted model.

    Returns:
        Mapping module prefix -> packed dict.
    """
    dec = model.model
    out = {}
    for li in range(len(dec.layers)):
        for sn in LINEAR_MODULE_NAMES:
            try:
                mod = dec.layers[li].get_submodule(sn)
            except AttributeError:
                continue
            if isinstance(mod, FoldInLinear):
                out[f"model.layers.{li}.{sn}"] = mod.export_folded()
    return out


def _set_requires_grad(model, predicate):
    """
    Enable gradients only for the parameters selected by `predicate`.

    Parameters:
        model: Module whose parameters are toggled.
        predicate: Callable(name) -> bool.

    Returns:
        List of the trainable parameters (in model order).
    """
    trainable = []
    for n, p in model.named_parameters():
        if predicate(n):
            p.requires_grad = True
            trainable.append(p)
        else:
            p.requires_grad = False
            p.grad = None
    return trainable


def _install_recon_hooks(model):
    """
    Register forward hooks that accumulate the per-module terms of L_RE.

    Parameters:
        model: Model containing FoldInLinear modules with teacher weights attached.

    Returns:
        reset: Clears the accumulated loss before a forward pass.
        total: Mean of the accumulated per-module terms (or None).
        remove: Removes the hooks.
    """
    state = {"sum": None, "count": 0, "collecting": True}
    handles = []

    def make_hook(mod):
        """
        Parameters:
            mod: FoldInLinear whose term is collected.

        Returns:
            Forward hook.
        """
        def post_hook(_module, inputs, output):
            """
            Add this module's L_RE term, computed from the module's actual output.

            Parameters:
                _module: Unused.
                inputs: Module inputs.
                output: Module output h_ad(x).
            """
            loss = mod.recon_loss(inputs[0], output)
            # not again in checkpoint recomputation
            if loss is None or not state["collecting"]:
                return
            state["sum"] = loss if state["sum"] is None else state["sum"] + loss
            state["count"] += 1
        return post_hook

    for mod in model.modules():
        if isinstance(mod, FoldInLinear) and mod._recon_teacher_w is not None:
            handles.append(mod.register_forward_hook(make_hook(mod)))

    def reset():
        """
        Clear the accumulated sum for the next training step.
        """
        state["sum"] = None
        state["count"] = 0
        state["collecting"] = True

    def total():
        """
        Returns:
            Mean per-module L_RE term since the last reset, or None.
        """
        state["collecting"] = False
        if state["count"] == 0 or state["sum"] is None:
            return None
        return state["sum"] / state["count"]

    def remove():
        """
        Remove every registered hook.
        """
        for h in handles:
            h.remove()

    return reset, total, remove


@torch.no_grad()
def _attach_recon_teachers(student, teacher):
    """
    Give every FoldInLinear a reference to the aligned model's weight W_a of the same module.

    Parameters:
        student: Model with FoldInLinear modules.
        teacher: Frozen aligned model with the same module names.

    Returns:
        Number of modules paired.
    """
    tmods = dict(teacher.named_modules())
    n = 0
    for name, mod in student.named_modules():
        if not isinstance(mod, FoldInLinear):
            continue
        t = tmods.get(name)
        w = getattr(t, "weight", None) if t is not None else None
        if w is None:
            continue
        mod._recon_teacher_w = w      # reference, not a copy
        n += 1
    print(f"[fold-in] L_RE targets: aligned-model weights of {n} modules")
    return n


@torch.no_grad()
def _clear_recon_teachers(model):
    """
    Drop the teacher-weight references from every FoldInLinear.

    Parameters:
        model: Model with FoldInLinear modules.
    """
    for mod in model.modules():
        if isinstance(mod, FoldInLinear):
            mod._recon_teacher_w = None


def _build_sched(optimizer, total_steps, warmup_steps, base_lr):
    """
    Linear warmup followed by cosine annealing.

    Parameters:
        optimizer: Optimizer to schedule.
        total_steps: Total number of optimizer steps.
        warmup_steps: Number of warmup steps.
        base_lr: Base learning rate (the cosine floor is 1% of it).

    Returns:
        The scheduler.
    """
    warm = WarmupLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cos = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps),
                            eta_min=base_lr * 0.01)
    return SequentialLR(optimizer, [warm, cos], milestones=[warmup_steps])


def train_fold_in(model, teacher, train_loader, hp: dict, device, teacher_device=None,
                  grad_ckpt: bool = False):
    """
    Train gamma and rho of every FoldInLinear with
    lambda_LM L_LM + lambda_KD L_KD + lambda_RE L_RE (eq. 20).

    Parameters:
        model: Adapted model (base model with FoldInLinear modules).
        teacher: Frozen aligned model.
        train_loader: Iterable of (input_ids, labels) batches.
        hp: Training hyper-parameters and loss weights of the configuration.
        device: Device of the adapted model.
        teacher_device: Device of the teacher (None = same device).
        grad_ckpt: Enable gradient checkpointing.
    """
    params = _set_requires_grad(model, lambda n: n.endswith(".rho") or n.endswith(".gamma"))
    if len(params) == 0:
        print("[fold-in] no tuning vectors; skipping.")
        return

    optimizer = torch.optim.AdamW(params, lr=hp["learning_rate"], weight_decay=hp["weight_decay"],
                                  betas=(0.9, 0.999), eps=1e-8)
    num_epochs = int(hp["epochs"])
    total_steps = max(1, len(train_loader) * num_epochs // hp["grad_accum"])
    warmup_steps = max(1, int(total_steps * hp["warmup_ratio"]))
    scheduler = _build_sched(optimizer, total_steps, warmup_steps, hp["learning_rate"])

    autocast = lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16)

    model.config.use_cache = False
    teacher.config.use_cache = False
    split_teacher = teacher_device is not None and not same_device(teacher_device, device)
    if split_teacher:
        print(f"[fold-in] teacher on {teacher_device}, adapted model on {device}")

    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("[fold-in] gradient checkpointing on")

    # teacher weights for L_RE
    _attach_recon_teachers(model, teacher)
    recon_reset, recon_total, recon_remove = _install_recon_hooks(model)

    print(f"[fold-in] trainable tensors={len(params)}, total_steps={total_steps}, "
          f"warmup={warmup_steps}, lr={hp['learning_rate']}")

    try:
        for epoch in range(num_epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f"[fold-in] epoch {epoch + 1}/{num_epochs}")
            optimizer.zero_grad()

            for bi, batch in enumerate(pbar):
                input_ids = batch[0].to(device)
                labels = batch[1].to(device)

                use_kd = hp["lambda_kd"] != 0
                if use_kd:
                    with torch.no_grad(), autocast():
                        if split_teacher:
                            ref_ids = input_ids.to(teacher_device, non_blocking=True)
                            ref_out = teacher(ref_ids, use_cache=False)
                            ref_logits = ref_out.logits.detach().to(device, non_blocking=True)
                            del ref_ids
                        else:
                            ref_out = teacher(input_ids, use_cache=False)
                            ref_logits = ref_out.logits.detach()
                    del ref_out

                with autocast():
                    recon_reset()
                    out = model(input_ids, labels=labels, use_cache=False)
                    lm = out.loss
                    recon = recon_total()
                    if recon is None:
                        recon = torch.zeros((), device=input_ids.device, dtype=torch.float32)
                    if use_kd:
                        vmin = min(out.logits.shape[-1], ref_logits.shape[-1])
                        log_p = F.log_softmax(out.logits[..., :vmin].float(), dim=-1)
                        p_ref = F.softmax(ref_logits[..., :vmin].float(), dim=-1)
                        if hp.get("kd_reduction", "batchmean") == "tokenmean":
                            distill = (F.kl_div(log_p, p_ref, reduction="sum")
                                       / (log_p.shape[0] * log_p.shape[1]))
                        else:
                            distill = F.kl_div(log_p, p_ref, reduction="batchmean")
                    else:
                        # no teacher forward
                        log_p = p_ref = None
                        distill = torch.zeros((), device=input_ids.device, dtype=torch.float32)

                    lm_w = hp["lambda_lm"] * lm
                    recon_w = hp["lambda_re"] * recon
                    distill_w = hp["lambda_kd"] * distill
                    loss = lm_w + distill_w + recon_w

                (loss / hp["grad_accum"]).backward()

                if (bi + 1) % hp["grad_accum"] == 0:
                    torch.nn.utils.clip_grad_norm_(params, hp["max_grad_norm"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                pbar.set_postfix({
                    "lm": f"{lm.item():.3f}({lm_w.item():.2f})",
                    "re": f"{recon.item():.3f}({recon_w.item():.2f})",
                    "kd": f"{distill.item():.3f}({distill_w.item():.2f})",
                    "tot": f"{loss.item():.2f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                })

                del out, log_p, p_ref
                if use_kd:
                    del ref_logits
    finally:
        recon_remove()
        _clear_recon_teachers(model)
    model.eval()
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        print(f"[fold-in] peak memory {device}: "
              f"{torch.cuda.max_memory_allocated(torch.device(device)) / 1024**3:.2f} GB")

