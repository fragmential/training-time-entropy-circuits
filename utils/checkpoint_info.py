import os

# Pythia: all models use batch size 1024 * seq_len 2048 = 2,097,152 tokens/step
PYTHIA_TOKENS_PER_STEP = 2_097_152

# Maps OLMo model keywords to their revision files (match most specific first)
OLMO_REVISION_FILES = {
    '7B': '7b_revisions.txt',
    '1B': '1b_revisions.txt',
}

def _parse_olmo_revisions(filepath: str) -> dict:
    """Parse an OLMo revisions file into a step -> token count (int) mapping."""
    step_to_tokens = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if 'step' in line and '-tokens' in line:
                step = int(line.split('-tokens')[0].split('step')[-1])
                tokens_str = line.split('-tokens')[-1].rstrip('B')
                step_to_tokens[step] = int(tokens_str) * 10**9
    return step_to_tokens

_olmo_cache = {}

def get_token_count(model_name: str, step_num: int) -> int:
    """Return the number of pretraining tokens at a given step for a model.

    For Pythia models: tokens = step_num * 2_097_152 (fixed batch size).
    For OLMo models: parsed from the revisions file.
    """
    if 'pythia' in model_name.lower():
        return step_num * PYTHIA_TOKENS_PER_STEP

    # OLMo: find the matching revision file by size suffix
    for suffix, fname in OLMO_REVISION_FILES.items():
        if model_name.endswith(suffix):
            if fname not in _olmo_cache:
                fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), fname)
                _olmo_cache[fname] = _parse_olmo_revisions(fpath)
            mapping = _olmo_cache[fname]
            if step_num in mapping:
                return mapping[step_num]
            raise ValueError(f"Step {step_num} not found in {fname}")

    raise ValueError(f"Unknown model family: {model_name}")
