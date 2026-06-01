"""Prefetch the e2e test models into the HF cache so test runs don't redownload.

Loads the FINAL/main checkpoint (revision=None) of each model used by
tests/test_e2e_all_hooks.py. Weights land under $HF_HOME and are reused on every
subsequent run. Run once on a node with network access:

    export HF_HOME="/projects/prjs1815/hf_cache"
    uv run python -m tests.prefetch_e2e_models
"""
from tests.e2e_models import E2E_MODELS, final_checkpoint
from utils.model_registry import get_model_config, load_model, load_tokenizer


def main():
    for name in E2E_MODELS:
        config = get_model_config(name)
        hf_model, revision, step = final_checkpoint(config)
        print(f"Prefetching {name} @ {revision} (step {step}) ...")
        load_tokenizer(config, revision=revision)
        load_model(config, hf_model, revision)
        print(f"  cached: {name}")
    print("Done.")


if __name__ == "__main__":
    main()
