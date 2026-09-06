import numpy as np


class Tensor:
    _id_counter = 0

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._children = _children
        self._op = _op
        self._backward = lambda: None
        Tensor._id_counter += 1
        self._id = Tensor._id_counter

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self._op})"

    # ---- helper for broadcasting grad back to original shape ----
    @staticmethod
    def _unbroadcast(grad, shape):
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, s in enumerate(shape):
            if s == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def _accumulate(self, g):
        if self.requires_grad:
            self.grad = self.grad + g

    # ---------------- ops ----------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op="add")

        def _backward():
            self._accumulate(Tensor._unbroadcast(out.grad, self.data.shape))
            other._accumulate(Tensor._unbroadcast(out.grad, other.data.shape))
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (other * -1.0 if isinstance(other, Tensor) else Tensor(-np.asarray(other, dtype=np.float64)))

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op="mul")

        def _backward():
            self._accumulate(Tensor._unbroadcast(out.grad * other.data, self.data.shape))
            other._accumulate(Tensor._unbroadcast(out.grad * self.data, other.data.shape))
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op="matmul")

        def _backward():
            g = out.grad
            if self.requires_grad:
                if self.data.ndim == 2:
                    self._accumulate(g @ np.swapaxes(other.data, -1, -2))
                else:
                    self._accumulate(g @ np.swapaxes(other.data, -1, -2))
            if other.requires_grad:
                if other.data.ndim == 2 and self.data.ndim == 2:
                    other._accumulate(self.data.T @ g)
                else:
                    other._accumulate(np.swapaxes(self.data, -1, -2) @ g)
        out._backward = _backward
        return out

    def transpose(self, *axes):
        out = Tensor(np.transpose(self.data, axes), requires_grad=self.requires_grad,
                     _children=(self,), _op="transpose")
        inv = np.argsort(axes)

        def _backward():
            self._accumulate(np.transpose(out.grad, inv))
        out._backward = _backward
        return out

    def reshape(self, *shape):
        orig_shape = self.data.shape
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad,
                     _children=(self,), _op="reshape")

        def _backward():
            self._accumulate(out.grad.reshape(orig_shape))
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad, _children=(self,), _op="sum")

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accumulate(np.broadcast_to(g, self.data.shape).copy())
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), requires_grad=self.requires_grad,
                     _children=(self,), _op="relu")

        def _backward():
            self._accumulate(out.grad * (self.data > 0))
        out._backward = _backward
        return out

    def gelu(self):
        # tanh approximation of GELU (standard, used in real ViT/BERT implementations)
        c = np.sqrt(2 / np.pi)
        x = self.data
        inner = c * (x + 0.044715 * x ** 3)
        t = np.tanh(inner)
        g = 0.5 * x * (1 + t)
        out = Tensor(g, requires_grad=self.requires_grad, _children=(self,), _op="gelu")

        def _backward():
            sech2 = 1 - t ** 2
            dinner_dx = c * (1 + 3 * 0.044715 * x ** 2)
            dg_dx = 0.5 * (1 + t) + 0.5 * x * sech2 * dinner_dx
            self._accumulate(out.grad * dg_dx)
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        x = self.data
        x_shift = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x_shift)
        s = e / np.sum(e, axis=axis, keepdims=True)
        out = Tensor(s, requires_grad=self.requires_grad, _children=(self,), _op="softmax")

        def _backward():
            g = out.grad
            dot = np.sum(g * s, axis=axis, keepdims=True)
            self._accumulate(s * (g - dot))
        out._backward = _backward
        return out

    def layernorm(self, gamma, beta, eps=1e-5):
        x = self.data
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        std = np.sqrt(var + eps)
        xhat = (x - mu) / std
        out_data = xhat * gamma.data + beta.data
        out = Tensor(out_data, requires_grad=True,
                     _children=(self, gamma, beta), _op="layernorm")
        D = x.shape[-1]

        def _backward():
            g = out.grad
            gamma._accumulate(Tensor._unbroadcast(g * xhat, gamma.data.shape))
            beta._accumulate(Tensor._unbroadcast(g, beta.data.shape))
            dxhat = g * gamma.data
            dvar = np.sum(dxhat * (x - mu) * -0.5 * std ** -3, axis=-1, keepdims=True)
            dmu = np.sum(dxhat * -1 / std, axis=-1, keepdims=True) + \
                  dvar * np.mean(-2 * (x - mu), axis=-1, keepdims=True)
            dx = dxhat / std + dvar * 2 * (x - mu) / D + dmu / D
            self._accumulate(dx)
        out._backward = _backward
        return out

    def cross_entropy(self, targets):
        """self: (N, C) logits. targets: (N,) int class indices (numpy array)."""
        x = self.data
        x_shift = x - np.max(x, axis=-1, keepdims=True)
        logsumexp = np.log(np.sum(np.exp(x_shift), axis=-1, keepdims=True))
        log_probs = x_shift - logsumexp
        N = x.shape[0]
        nll = -log_probs[np.arange(N), targets]
        loss_val = nll.mean()
        out = Tensor(loss_val, requires_grad=True, _children=(self,), _op="cross_entropy")

        def _backward():
            probs = np.exp(log_probs)
            grad = probs.copy()
            grad[np.arange(N), targets] -= 1
            grad /= N
            self._accumulate(grad * out.grad)
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()

        def build(t):
            if id(t) not in visited:
                visited.add(id(t))
                for c in t._children:
                    build(c)
                topo.append(t)
        build(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            t._backward()


def numerical_gradient_check():
    """Sanity check the autograd engine against finite differences before
    trusting it for anything -- this is the single most important cell in
    this file."""
    rng = np.random.default_rng(0)
    A = Tensor(rng.normal(size=(4, 5)), requires_grad=True)
    B = Tensor(rng.normal(size=(5, 3)), requires_grad=True)
    gamma = Tensor(rng.normal(size=(3,)) * 0 + 1, requires_grad=True)
    beta = Tensor(rng.normal(size=(3,)), requires_grad=True)

    def forward():
        h = (A @ B).gelu()
        h = h.layernorm(gamma, beta)
        s = h.softmax(axis=-1)
        return s.sum()

    out = forward()
    out.backward()
    analytic = A.grad.copy()

    eps = 1e-5
    numeric = np.zeros_like(A.data)
    for i in range(A.data.shape[0]):
        for j in range(A.data.shape[1]):
            orig = A.data[i, j]
            A.data[i, j] = orig + eps
            A.grad = np.zeros_like(A.data); B.grad = np.zeros_like(B.data)
            gamma.grad = np.zeros_like(gamma.data); beta.grad = np.zeros_like(beta.data)
            plus = forward().data
            A.data[i, j] = orig - eps
            minus = forward().data
            A.data[i, j] = orig
            numeric[i, j] = (plus - minus) / (2 * eps)

    max_err = np.max(np.abs(analytic - numeric))
    denom = np.abs(numeric)
    denom[denom < 1e-6] = 1e-6  # avoid spurious blow-up near-zero entries
    rel_err = np.max(np.abs(analytic - numeric) / denom)
    return max_err, rel_err


if __name__ == "__main__":
    max_err, rel_err = numerical_gradient_check()
    print(f"Gradient check: max abs error={max_err:.2e}, max rel error={rel_err:.2e}")
    assert max_err < 1e-6, f"Autograd engine FAILED gradient check (max_abs_err={max_err:.2e})"
    print("Autograd engine verified correct (matmul, gelu, layernorm, softmax chained).")
