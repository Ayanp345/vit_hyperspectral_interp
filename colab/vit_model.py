import numpy as np
from micrograd_tensor import Tensor


def init_linear(fan_in, fan_out, rng):
    scale = np.sqrt(2.0 / fan_in)
    W = Tensor(rng.normal(0, scale, size=(fan_in, fan_out)), requires_grad=True)
    b = Tensor(np.zeros(fan_out), requires_grad=True)
    return W, b


class MultiHeadSelfAttention:
    def __init__(self, embed_dim, n_heads, rng):
        assert embed_dim % n_heads == 0
        self.embed_dim, self.n_heads = embed_dim, n_heads
        self.head_dim = embed_dim // n_heads
        self.Wq, self.bq = init_linear(embed_dim, embed_dim, rng)
        self.Wk, self.bk = init_linear(embed_dim, embed_dim, rng)
        self.Wv, self.bv = init_linear(embed_dim, embed_dim, rng)
        self.Wo, self.bo = init_linear(embed_dim, embed_dim, rng)
        self.last_attn_weights = None  # (n_heads, seq, seq) — for interpretability

    def params(self):
        return [self.Wq, self.bq, self.Wk, self.bk, self.Wv, self.bv, self.Wo, self.bo]

    def __call__(self, x):
        """x: Tensor (seq_len, embed_dim). Single "batch item" at a time for
        simplicity (this model is small enough that per-sample loops are
        fine on CPU)."""
        seq_len = x.shape[0]
        Q = x @ self.Wq + self.bq
        K = x @ self.Wk + self.bk
        V = x @ self.Wv + self.bv

        Qh = Q.reshape(seq_len, self.n_heads, self.head_dim).transpose(1, 0, 2)
        Kh = K.reshape(seq_len, self.n_heads, self.head_dim).transpose(1, 0, 2)
        Vh = V.reshape(seq_len, self.n_heads, self.head_dim).transpose(1, 0, 2)

        scale = 1.0 / np.sqrt(self.head_dim)
        scores = (Qh @ Kh.transpose(0, 2, 1)) * scale         # (heads, seq, seq)
        attn = scores.softmax(axis=-1)
        self.last_attn_weights = attn.data.copy()

        out_h = attn @ Vh                                      # (heads, seq, head_dim)
        out = out_h.transpose(1, 0, 2).reshape(seq_len, self.embed_dim)
        out = out @ self.Wo + self.bo
        return out


class MLPBlock:
    def __init__(self, embed_dim, hidden_dim, rng):
        self.W1, self.b1 = init_linear(embed_dim, hidden_dim, rng)
        self.W2, self.b2 = init_linear(hidden_dim, embed_dim, rng)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def __call__(self, x):
        h = (x @ self.W1 + self.b1).gelu()
        return h @ self.W2 + self.b2


class LayerNormModule:
    def __init__(self, dim):
        self.gamma = Tensor(np.ones(dim), requires_grad=True)
        self.beta = Tensor(np.zeros(dim), requires_grad=True)

    def params(self):
        return [self.gamma, self.beta]

    def __call__(self, x):
        return x.layernorm(self.gamma, self.beta)


class EncoderBlock:
    def __init__(self, embed_dim, n_heads, mlp_hidden, rng):
        self.ln1 = LayerNormModule(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, rng)
        self.ln2 = LayerNormModule(embed_dim)
        self.mlp = MLPBlock(embed_dim, mlp_hidden, rng)

    def params(self):
        return self.ln1.params() + self.attn.params() + self.ln2.params() + self.mlp.params()

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class HyperspectralViT:
    def __init__(self, n_bands, patch_size, grid_size, n_classes,
                 embed_dim=48, n_heads=4, n_layers=3, mlp_hidden=96, seed=0):
        rng = np.random.default_rng(seed)
        self.patch_size = patch_size
        self.grid_size = grid_size          # e.g. 4 -> 4x4=16 patches
        self.n_bands = n_bands
        self.embed_dim = embed_dim
        patch_dim = patch_size * patch_size * n_bands
        n_patches = grid_size * grid_size

        self.Wpatch, self.bpatch = init_linear(patch_dim, embed_dim, rng)
        self.pos_embed = Tensor(rng.normal(0, 0.02, size=(n_patches + 1, embed_dim)),
                                 requires_grad=True)
        self.cls_token = Tensor(rng.normal(0, 0.02, size=(1, embed_dim)), requires_grad=True)

        self.blocks = [EncoderBlock(embed_dim, n_heads, mlp_hidden, rng) for _ in range(n_layers)]
        self.ln_final = LayerNormModule(embed_dim)
        self.Whead, self.bhead = init_linear(embed_dim, n_classes, rng)

        self._activation_cache = {}  # populated during forward() for interpretability

    def params(self):
        p = [self.Wpatch, self.bpatch, self.pos_embed, self.cls_token]
        for b in self.blocks:
            p += b.params()
        p += self.ln_final.params() + [self.Whead, self.bhead]
        return p

    def patchify(self, cube):
        """cube: numpy (H, W, bands) -> (n_patches, patch_dim) numpy array."""
        H, W, B = cube.shape
        ps, gs = self.patch_size, self.grid_size
        patches = np.zeros((gs * gs, ps * ps * B))
        idx = 0
        for i in range(gs):
            for j in range(gs):
                block = cube[i * ps:(i + 1) * ps, j * ps:(j + 1) * ps, :]
                patches[idx] = block.reshape(-1)
                idx += 1
        return patches

    def forward(self, cube, cache=True):
        patches_np = self.patchify(cube)                      # (n_patches, patch_dim)
        patches = Tensor(patches_np, requires_grad=False)
        embedded = patches @ self.Wpatch + self.bpatch          # (n_patches, embed_dim)

        tokens_data = np.concatenate([self.cls_token.data, embedded.data], axis=0)
        tokens = Tensor(tokens_data, requires_grad=True)
        # rewire graph so gradients flow to cls_token and embedded:
        tokens._children = (self.cls_token, embedded)
        tokens._op = "concat_cls"

        def _backward():
            self.cls_token._accumulate(tokens.grad[:1])
            embedded._accumulate(tokens.grad[1:])
        tokens._backward = _backward

        x = tokens + self.pos_embed

        if cache:
            self._activation_cache = {"patch_embed": embedded.data.copy(),
                                       "layer_outputs": [], "attn_weights": []}

        for li, block in enumerate(self.blocks):
            x = block(x)
            if cache:
                self._activation_cache["layer_outputs"].append(x.data.copy())
                self._activation_cache["attn_weights"].append(block.attn.last_attn_weights.copy())

        x_final = self.ln_final(x)
        cls_repr = x_final.reshape(x_final.shape[0], self.embed_dim)
        # slice CLS token (index 0) -- implement via matmul with selection vector
        sel = np.zeros((1, x_final.shape[0]))
        sel[0, 0] = 1.0
        sel_t = Tensor(sel, requires_grad=False)
        cls_vec = sel_t @ cls_repr                              # (1, embed_dim)
        logits = cls_vec @ self.Whead + self.bhead               # (1, n_classes)

        if cache:
            self._activation_cache["cls_final"] = cls_vec.data.copy()
            self._activation_cache["logits"] = logits.data.copy()

        return logits


def sgd_step(params, lr):
    for p in params:
        p.data -= lr * p.grad
        p.grad = np.zeros_like(p.data)


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * p.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (p.grad ** 2)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
            p.grad = np.zeros_like(p.data)
