"""The kernel is a torch.library custom op: opcheck-clean and torch.compile-safe."""

import math
import sys

import pytest
import torch

from fa_rdna3 import flash_attention

DEVICE = "cuda"


def test_opcheck():
    q = torch.randn(2, 4, 128, 64, device=DEVICE, dtype=torch.float16, requires_grad=True)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    args = (q, k, v, True, 1.0 / math.sqrt(64), -1, -1, 0.0, None, None, 0.0, 0)
    # Skip the fp64 gradcheck utilities (the kernel is fp16-only); the schema,
    # fake-tensor and autograd-registration checks are the meaningful ones here.
    torch.library.opcheck(
        torch.ops.fa_rdna3.flash_fwd, args,
        test_utils=("test_schema", "test_faketensor", "test_autograd_registration"))


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="torch.compile unsupported on Python 3.14+")
def test_torch_compile_matches_eager():
    q = torch.randn(2, 4, 256, 64, device=DEVICE, dtype=torch.float16, requires_grad=True)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = 1.0 / math.sqrt(64)

    def f(q, k, v):
        return flash_attention(q, k, v, causal=True, softmax_scale=scale).square().sum()

    eager = f(q, k, v)
    eager.backward()
    eager_dq = q.grad.clone()

    q.grad = None
    compiled = torch.compile(f, fullgraph=True)(q, k, v)
    compiled.backward()

    torch.testing.assert_close(compiled, eager, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(q.grad, eager_dq, atol=2e-3, rtol=2e-3)
