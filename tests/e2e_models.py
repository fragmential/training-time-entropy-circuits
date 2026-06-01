"""Shared model list + hook patterns for the all-hooks e2e tests.

Kept separate so prefetch and the test module agree on exactly which models /
checkpoints must be cached.
"""

# Smallest model per family. OLMo-2's smallest is 1B (no tiny variant exists).
PYTHIA = "EleutherAI/pythia-14m"
OLMO = "allenai/OLMo-2-0425-1B"
E2E_MODELS = [PYTHIA, OLMO]

# Hook patterns covering every kind the family exposes. `+G` adds the gradient
# (backward) path; per-OV-head (`attn.head*`) and the kwargs-called `attn.in`
# boundary are the paths that only a real forward/backward pass exercises.
_COMMON = [
    "blk*.up+G", "blk*.down+G",                       # MLP input A + G
    "blk*.attn.head*+G",                              # per-OV-head slice A + G
    "blk*.attn.in", "blk*.attn.out", "blk*.mlp.out",  # sub-block boundaries
    "before_final_norm+G", "after_final_norm+G",      # residual stream A + G
]
PYTHIA_HOOKS = list(_COMMON)
OLMO_HOOKS = _COMMON + [
    "blk*.gate+G",                       # OLMo-2 third MLP projection
    "blk*.attn.raw_out", "blk*.mlp.in",  # OLMo-2-only boundaries
]

HOOKS_BY_MODEL = {PYTHIA: PYTHIA_HOOKS, OLMO: OLMO_HOOKS}


def final_checkpoint(config):
    """(hf_model, revision, step) for the model's last/final checkpoint."""
    from utils.model_registry import get_checkpoint_schedule
    schedule = get_checkpoint_schedule(config)
    step, revision, hf_model = max(schedule, key=lambda t: t[0])
    return hf_model, revision, step
