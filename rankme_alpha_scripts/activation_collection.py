import torch
import torch.nn as nn

HOOK_LOCATION_BY_METHOD = {"identity": "identity", "hf_hidden_states": "hidden_states_last", "kfac": "kfac"}


def replace_output_head_with_identity(model) -> str:
    """
    Replace model output head with Identity for logit-space extraction.
    Returns the replaced attribute name (or generic label).
    """
    for attr in ("lm_head", "embed_out"):
        if hasattr(model, attr):
            setattr(model, attr, nn.Identity())
            return attr

    if hasattr(model, "set_output_embeddings"):
        model.set_output_embeddings(nn.Identity())
        return "output_embeddings"

    raise ValueError("Could not locate output head (lm_head/embed_out/output_embeddings) for identity collection")


def collect_last_token_activations(model, input_ids, attention_mask, collection_method: str):
    if collection_method == "kfac":
        raise NotImplementedError(
            "collection_method='kfac' is reserved but not implemented yet. "
            "Use 'identity' or 'hf_hidden_states' for now."
        )

    if collection_method == "hf_hidden_states":
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        states = outputs.hidden_states[-1]
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        states = outputs.logits

    last_indices = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
    return states[batch_indices, last_indices, ...]
