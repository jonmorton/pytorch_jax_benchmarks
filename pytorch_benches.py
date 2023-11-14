import torch
from common import Bench

torch.set_float32_matmul_precision("high")


class PytorchBench(Bench):
    def run(self):
        loss = self.run_internal()
        loss.backward()
        torch.cuda.synchronize()

    def run_internal(self):
        pass


class Resnet50(PytorchBench):
    def setup(self, batch_size):
        import torchvision

        self.module = torch.compile(torchvision.models.resnet50().to("cuda"))
        self.input = torch.randn(batch_size, 3, 224, 224).to("cuda")

    def run_internal(self):
        return self.module(self.input).mean()


class SelfAttn(PytorchBench):
    def __init__(self, dim, seq_len, dtype=torch.float32):
        self.dim = dim
        self.seq_len = seq_len
        self.dtype = dtype

    def setup(self, batch_size):
        from pytorch_transformer import CausalSelfAttention

        self.module = torch.compile(
            CausalSelfAttention(self.dim, self.seq_len).to("cuda")
        )
        self.input = (
            torch.randn(batch_size, self.seq_len, self.dim).to("cuda").to(self.dtype)
        )

    def run_internal(self):
        return self.module(self.input).mean()


PYTORCH_BENCHES = {
    "resnet50": Resnet50,
    "attn_seq1024_dim512_tf32": lambda: SelfAttn(1024, 512),
    "attn_seq2048_dim256_tf32": lambda: SelfAttn(2048, 256),
    "attn_seq1024_dim512_f16": lambda: SelfAttn(1024, 512, torch.float16),
}
