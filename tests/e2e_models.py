"""Shared model list + hook patterns for the all-hooks e2e tests.

Kept separate so prefetch and the test module agree on exactly which models /
checkpoints must be cached.
"""

# Smallest model per family. OLMo-2's smallest is 1B (no tiny variant exists).
PYTHIA = "EleutherAI/pythia-14m"
OLMO = "allenai/OLMo-2-0425-1B"
E2E_MODELS = [PYTHIA, OLMO]

# Leaf patterns covering every kind the family exposes. `:both`/:grads/preset:kfac
# drive the backward path; per-OV-head (`attn.head*.slice`) and the kwargs-called
# `attn.in` boundary are the paths only a real forward/backward pass exercises.
_COMMON = [
    "preset:kfac",                                     # MLP K-FAC: in:acts + out:grads
    "blk*.attn.head*.slice:both",                      # per-OV-head slice acts + grads
    "blk*.attn.in", "blk*.attn.out", "blk*.mlp.out",   # sub-block boundaries (acts)
    "before_final_norm:both", "after_final_norm:both", # residual stream acts + grads
]
PYTHIA_HOOKS = list(_COMMON)
OLMO_HOOKS = _COMMON + [
    "blk*.attn.raw_out", "blk*.mlp.in",  # OLMo-2-only boundaries (preset:kfac already adds gate)
]

HOOKS_BY_MODEL = {PYTHIA: PYTHIA_HOOKS, OLMO: OLMO_HOOKS}


def final_checkpoint(config):
    """(hf_model, revision, step) for the model's last/final checkpoint."""
    from utils.model_registry import get_checkpoint_schedule
    schedule = get_checkpoint_schedule(config)
    step, revision, hf_model = max(schedule, key=lambda t: t[0])
    return hf_model, revision, step
