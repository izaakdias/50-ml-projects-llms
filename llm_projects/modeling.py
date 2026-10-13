from __future__ import annotations

import numpy as np
import re

from .core import CORPUS, IOI_PROMPTS, IOI_TARGETS, token_category


def load_gpt2_bundle(model_name: str) -> dict:
    """Extract a compact set of internals from a GPT-2-compatible HF checkpoint."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements-models.txt to use --model distilgpt2") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    texts = IOI_PROMPTS + CORPUS
    encoded = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=16, return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping").cpu().numpy()
    encoded = {key: value.to(device) for key, value in encoded.items()}
    captures: list[dict] = [dict() for _ in model.transformer.h]
    handles = []
    for layer_index, block in enumerate(model.transformer.h):
        def capture_qkv(module, inputs, output, layer_index=layer_index):
            captures[layer_index]["qkv"] = output.detach()
        def capture_attn_update(module, inputs, output, layer_index=layer_index):
            value = output[0] if isinstance(output, tuple) else output
            captures[layer_index]["attn_update"] = value.detach()
        def capture_pre(module, inputs, output, layer_index=layer_index):
            captures[layer_index]["mlp_pre"] = output.detach()
        def capture_act(module, inputs, output, layer_index=layer_index):
            captures[layer_index]["mlp_act"] = output.detach()
        def capture_mlp_update(module, inputs, output, layer_index=layer_index):
            captures[layer_index]["mlp_update"] = output.detach()
        handles.extend([
            block.attn.c_attn.register_forward_hook(capture_qkv),
            block.attn.register_forward_hook(capture_attn_update),
            block.mlp.c_fc.register_forward_hook(capture_pre),
            block.mlp.act.register_forward_hook(capture_act),
            block.mlp.c_proj.register_forward_hook(capture_mlp_update),
        ])
    with torch.no_grad():
        output = model(**encoded, output_hidden_states=True, output_attentions=True, use_cache=False)
    for handle in handles:
        handle.remove()

    input_ids = encoded["input_ids"].cpu().numpy()
    hidden_states = np.stack([x.detach().cpu().numpy() for x in output.hidden_states])
    attentions = np.stack([x.detach().cpu().numpy() for x in output.attentions])
    n_layers, batch, seq_len, d_model = len(captures), input_ids.shape[0], input_ids.shape[1], model.config.n_embd
    n_heads = model.config.n_head
    d_head = d_model // n_heads
    q_all, k_all, v_all, score_all, contribution_all = [], [], [], [], []
    pre_all, act_all, update_all, attn_update_all, mlp_biases, qkv_weights, wo_weights, w1_weights, w2_weights = [], [], [], [], [], [], [], [], []
    causal = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    attention_mask = encoded["attention_mask"].cpu().numpy().astype(bool)
    for layer_index, block in enumerate(model.transformer.h):
        qkv_tensor = captures[layer_index]["qkv"]
        qkv = qkv_tensor.detach().cpu().numpy().reshape(batch, seq_len, 3, n_heads, d_head)
        q, k, v = [qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3)]
        raw = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d_head)
        mask = causal[None, None, :, :] | (~attention_mask[:, None, None, :])
        raw = np.where(mask, -1e9, raw)
        head_values = attentions[layer_index] @ v
        weight = block.attn.c_proj.weight.detach().cpu().numpy()
        contribs = np.stack([head_values[:, h] @ weight[h*d_head:(h+1)*d_head, :] for h in range(n_heads)], axis=1)
        q_all.append(q); k_all.append(k); v_all.append(v); score_all.append(raw); contribution_all.append(contribs)
        pre_all.append(captures[layer_index]["mlp_pre"].cpu().numpy())
        act_all.append(captures[layer_index]["mlp_act"].cpu().numpy())
        update_all.append(captures[layer_index]["mlp_update"].cpu().numpy())
        attn_update_all.append(captures[layer_index]["attn_update"].cpu().numpy())
        mlp_biases.append(block.mlp.c_proj.bias.detach().cpu().numpy())
        qkv_weights.append(tuple(block.attn.c_attn.weight.detach().cpu().numpy()[:, i*d_model:(i+1)*d_model] for i in range(3)))
        wo_weights.append(weight.reshape(n_heads, d_head, d_model))
        w1_weights.append(block.mlp.c_fc.weight.detach().cpu().numpy())
        w2_weights.append(block.mlp.c_proj.weight.detach().cpu().numpy())
    embeddings = model.transformer.wte.weight.detach().cpu().numpy()
    unembedding = model.lm_head.weight.detach().cpu().numpy().T
    final_pre_norm = hidden_states[-2] + attn_update_all[-1] + update_all[-1]
    token_strings = [tokenizer.convert_ids_to_tokens(row.tolist()) for row in input_ids]
    token_categories = []
    subject_positions = []
    for text_index, (text, row_offsets) in enumerate(zip(texts, offsets)):
        spans = [(match.start(), match.end(), token_category(match.group(0))) for match in re.finditer(r"[\w]+(?:['’][\w]+)?", text, flags=re.UNICODE)]
        labels = []
        for start, end in row_offsets:
            overlapping = [label for left, right, label in spans if end > start and max(start, left) < min(end, right)]
            labels.append(overlapping[0] if overlapping else "other")
        token_categories.append(labels)
        if text_index < len(IOI_PROMPTS):
            subject = "leo" if "leo gave" in text.lower() else "maya"
            occurrences = list(re.finditer(rf"\b{subject}\b", text, flags=re.IGNORECASE))
            subject_span = occurrences[-1].span()
            positions = [i for i, (start, end) in enumerate(row_offsets) if end > start and max(start, subject_span[0]) < min(end, subject_span[1])]
            subject_positions.append(positions[0])
        else:
            subject_positions.append(-1)
    ioi_target_ids = [tokenizer.encode(" " + target, add_special_tokens=False)[0] for target in IOI_TARGETS]
    return {
        "kind": "huggingface_gpt2", "model_name": model_name, "tokenizer": tokenizer,
        "tokens": input_ids, "token_strings": token_strings, "token_categories": token_categories,
        "attention_mask": attention_mask, "last_positions": attention_mask.sum(axis=1).astype(int) - 1,
        "subject_positions": np.asarray(subject_positions), "ioi_target_ids": np.asarray(ioi_target_ids),
        "vocab": [tokenizer.convert_ids_to_tokens(i) for i in range(len(tokenizer))], "embeddings": embeddings,
        "hidden_states": hidden_states, "q": np.stack(q_all), "k": np.stack(k_all), "v": np.stack(v_all),
        "attention_scores": np.stack(score_all), "attention": attentions, "attention_contribs": np.stack(contribution_all),
        "attention_updates": np.stack(attn_update_all), "final_pre_norm": final_pre_norm,
        "final_norm_weight": model.transformer.ln_f.weight.detach().cpu().numpy(),
        "final_norm_bias": model.transformer.ln_f.bias.detach().cpu().numpy(), "final_norm_eps": model.transformer.ln_f.eps,
        "mlp_scale": 1.0, "mlp_biases": mlp_biases,
        "mlp_pre": np.stack(pre_all), "mlp_act": np.stack(act_all), "mlp_update": np.stack(update_all),
        "qkv_weights": qkv_weights, "wo_weights": wo_weights, "w1_weights": w1_weights, "w2_weights": w2_weights,
        "unembedding": unembedding, "logits": output.logits.detach().cpu().numpy(),
        "prompt_texts": texts,
    }
