import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax, nn

ortho_init = nn.initializers.orthogonal()


def make_causal_mask(seq_len):
    idxs = jnp.arange(seq_len, dtype=jnp.int32)
    mask = jnp.greater_equal(
        jnp.expand_dims(idxs, axis=-1), jnp.expand_dims(idxs, axis=-2)
    )
    return mask


class CausalSelfAttention(eqx.Module):
    nhead: int
    dim_head: int
    Wqkv: jax.Array
    Wout: jax.Array
    bias: jax.Array
    causal_mask: jax.Array = eqx.field(static=True)

    def __init__(self, key, dim, seq_len, dim_head=64):
        self.nhead = dim // dim_head
        self.dim_head = dim_head
        k1, k2 = jax.random.split(key)
        self.Wqkv = ortho_init(k1, (dim, dim * 3), jnp.float32)
        self.Wout = ortho_init(k2, (dim, dim), jnp.float32)
        self.bias = jnp.zeros(dim)
        self.causal_mask = make_causal_mask(seq_len)

    def params(self):
        return [self.Wqkv, self.Wout, self.bias]

    def __call__(self, x):
        seq_len = x.shape[-2]
        qkv = x @ self.Wqkv
        qkv = jnp.reshape(qkv, (-1, 3, self.nhead, self.dim_head))
        qkv = jnp.swapaxes(qkv, 0, 2)  # (nhead, 3, seq, dim_head)
        q, k, v = [lax.index_in_dim(qkv, i, -3, keepdims=False) for i in range(3)]

        logits = (q @ jnp.swapaxes(k, -1, -2)) / jnp.sqrt(self.dim_head)
        logits = jnp.where(
            self.causal_mask[:seq_len, :seq_len], logits, jnp.finfo(logits.dtype).min
        )
        probs = nn.softmax(logits, axis=-1)
        attn_outs = probs @ v
        attn_outs = jnp.reshape(
            jnp.swapaxes(attn_outs, 0, 1), (-1, self.nhead * self.dim_head)
        )

        proj_out = attn_outs @ self.Wout
        return proj_out + self.bias
