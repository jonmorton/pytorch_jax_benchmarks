import equinox as eqx
import jax
from common import Bench

jax.default_matmul_precision("tensorfloat32")


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def loss_fn_with_state(model, x, state):
    pred, state = jax.vmap(
        model, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(
        x,
        state,
    )
    return pred.mean(), state


@eqx.filter_jit
@eqx.filter_value_and_grad
def loss_fn(model, x):
    pred = jax.vmap(model)(x)
    return pred.mean()


class JaxBench(Bench):
    pass


class Resnet50(JaxBench):
    def setup(self, batch_size):
        from jax_resnet import resnet50

        self.module = resnet50()
        self.input = jax.random.normal(
            jax.random.PRNGKey(42), (batch_size, 3, 224, 224), dtype=jax.numpy.float32
        )
        self.state = eqx.nn.State(self.module)

    def run(self):
        (val, self.state), _ = loss_fn_with_state(self.module, self.input, self.state)
        val.block_until_ready()


class SelfAttn(JaxBench):
    def __init__(self, dim, seq_len, dtype=jax.numpy.float32):
        self.dim = dim
        self.seq_len = seq_len
        self.dtype = dtype

    def setup(self, batch_size):
        from jax_transformer import CausalSelfAttention

        self.module = CausalSelfAttention(jax.random.PRNGKey(0), self.dim, self.seq_len)
        self.input = jax.random.normal(
            jax.random.PRNGKey(42),
            (batch_size, self.seq_len, self.dim),
            dtype=self.dtype,
        )

    def run(self):
        val, grad = loss_fn(self.module, self.input)
        val.block_until_ready()


JAX_BENCHES = {
    "resnet50": Resnet50,
    "attn_seq1024_dim512_tf32": lambda: SelfAttn(1024, 512),
    "attn_seq2048_dim256_tf32": lambda: SelfAttn(2048, 256),
    "attn_seq1024_dim512_f16": lambda: SelfAttn(1024, 512, jax.numpy.float16),
}
