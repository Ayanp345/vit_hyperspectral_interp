import numpy as np
from micrograd_tensor import Tensor


def logit_lens(model, cube_norm):
    """Run the model once, capturing the residual stream at every layer, and
    apply the FINAL classification head (Whead, bhead) + final LayerNorm to
    each intermediate layer's CLS-token representation. Returns
    (n_layers, n_classes) array of logits — one row per layer "as if that
    were the last layer."""
    logits_out = model.forward(cube_norm, cache=True)  # sets model._activation_cache
    cache = model._activation_cache
    layer_outputs = cache["layer_outputs"]  # list of (seq_len, embed_dim) arrays

    per_layer_logits = []
    for layer_repr in layer_outputs:
        cls_raw = layer_repr[0:1, :]  # CLS token, pre-final-LN
        # apply the SAME final LayerNorm + head used at the true output,
        # exactly as the logit-lens technique prescribes (reuse trained
        # readout weights, not a freshly fit probe)
        gamma, beta = model.ln_final.gamma.data, model.ln_final.beta.data
        mu = cls_raw.mean(axis=-1, keepdims=True)
        var = cls_raw.var(axis=-1, keepdims=True)
        normed = (cls_raw - mu) / np.sqrt(var + 1e-5) * gamma + beta
        logits = normed @ model.Whead.data + model.bhead.data
        per_layer_logits.append(logits[0])
    return np.array(per_layer_logits), logits_out.data[0]


def attention_rollout(model, cube_norm):
    """Abnar & Zuidema (2020) attention rollout: average attention heads per
    layer, add identity (residual contribution), row-normalize, then
    multiply layer matrices together. Returns (n_patches,) relevance scores
    for the CLS token's final prediction over the spatial patches (CLS token
    itself excluded from the reported vector)."""
    model.forward(cube_norm, cache=True)
    attn_list = model._activation_cache["attn_weights"]  # list of (heads, seq, seq)

    seq_len = attn_list[0].shape[-1]
    rollout = np.eye(seq_len)
    for attn in attn_list:
        avg_heads = attn.mean(axis=0)                    # (seq, seq)
        avg_heads = avg_heads + np.eye(seq_len)           # residual correction
        avg_heads = avg_heads / avg_heads.sum(axis=-1, keepdims=True)
        rollout = avg_heads @ rollout

    cls_relevance = rollout[0, 1:]  # CLS row, excluding CLS-to-CLS entry
    return cls_relevance / (cls_relevance.sum() + 1e-8)


def spectral_band_attribution(model, cube_norm, target_class, band_std):
    """Gradient x input saliency w.r.t. the RAW (normalized) input cube for
    a given target class logit, aggregated to one score per spectral band.

    Implementation note: since patchify() is a fixed reshape (not a Tensor
    op with tracked gradients in this small engine), we compute the gradient
    of the target logit w.r.t. the patch-embedding INPUT tensor (`patches`)
    manually by re-running the forward pass with patches as a leaf Tensor,
    then reshape that gradient back into (H, W, bands) and multiply by the
    input to get gradient x input attribution -- then average |attr| across
    all spatial positions for each band to get a single per-band importance
    score, and divide by band_std to undo the earlier standardization so the
    result is interpretable in original reflectance units.
    """
    H, W, B = cube_norm.shape
    ps, gs = model.patch_size, model.grid_size
    patches_np = model.patchify(cube_norm)
    patches = Tensor(patches_np, requires_grad=True)

    embedded = patches @ model.Wpatch + model.bpatch
    tokens_data = np.concatenate([model.cls_token.data, embedded.data], axis=0)
    tokens = Tensor(tokens_data, requires_grad=True)
    tokens._children = (model.cls_token, embedded)

    def _tok_backward():
        model.cls_token._accumulate(tokens.grad[:1])
        embedded._accumulate(tokens.grad[1:])
    tokens._backward = _tok_backward

    x = tokens + model.pos_embed
    for block in model.blocks:
        x = block(x)
    x_final = model.ln_final(x)
    cls_repr = x_final.reshape(x_final.shape[0], model.embed_dim)
    sel = np.zeros((1, x_final.shape[0])); sel[0, 0] = 1.0
    cls_vec = Tensor(sel, requires_grad=False) @ cls_repr
    logits = cls_vec @ model.Whead + model.bhead

    target = Tensor(np.zeros_like(logits.data), requires_grad=False)
    target.data[0, target_class] = 1.0
    selected_logit = (logits * target).sum()
    selected_logit.backward()

    grad_patches = patches.grad                      # (n_patches, patch_dim)
    grad_x_input = grad_patches * patches_np           # gradient x input

    # reshape back to (H, W, B) and aggregate |attribution| per band
    band_scores = np.zeros(B)
    idx = 0
    counts = np.zeros(B)
    for i in range(gs):
        for j in range(gs):
            block_grad = grad_x_input[idx].reshape(ps, ps, B)
            band_scores += np.abs(block_grad).sum(axis=(0, 1))
            counts += ps * ps
            idx += 1
    band_scores = band_scores / counts
    band_scores = band_scores / (band_std + 1e-8)  # undo standardization scale
    return band_scores, logits.data[0]
