import pickle
from functools import partial
from typing import Self

import jax
from flax import nnx
from huggingface_hub import hf_hub_download


class LPIPSWeight(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        conv = partial(
            nnx.Conv,
            out_features=1,
            kernel_size=(1, 1),
            strides=None,
            padding=0,
            use_bias=False,
        )

        self.weights = {
            "relu1_2": conv(in_features=64, rngs=rngs),
            "relu2_2": conv(in_features=128, rngs=rngs),
            "relu3_3": conv(in_features=256, rngs=rngs),
            "relu4_3": conv(in_features=512, rngs=rngs),
            "relu5_3": conv(in_features=512, rngs=rngs),
        }

    def __call__(self, x: jax.Array, feature: str) -> jax.Array:
        return self.weights[feature](x)

    @classmethod
    def load_pretrained(cls) -> Self:
        weights_file = hf_hub_download(
            repo_id="lweitkamp/lpips-nnx", filename="linear.pkl"
        )
        weights = pickle.load(open(weights_file, "rb"))

        graphdef, _ = nnx.split(cls(nnx.Rngs(0)))
        return nnx.merge(graphdef, weights)
