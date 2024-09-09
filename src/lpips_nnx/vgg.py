import pickle
from functools import partial
from typing import NamedTuple, Self

import jax
import jax.numpy as jnp
from flax import nnx
from flax.linen import max_pool
from huggingface_hub import hf_hub_download


class LPIPSfeatures(NamedTuple):
    relu1_2: jax.Array
    relu2_2: jax.Array
    relu3_3: jax.Array
    relu4_3: jax.Array
    relu5_3: jax.Array


class ConvBlock(nnx.Module):
    def __init__(self, dim_in: int, dim_out: int, n_layers: int, rngs: nnx.Rngs):
        dims = [(dim_in, dim_out)] + [(dim_out, dim_out)] * (n_layers - 1)
        conv = partial(nnx.Conv, kernel_size=(3, 3), padding="SAME")
        self.layers = [conv(dim_in, dim_out, rngs=rngs) for dim_in, dim_out in dims]

    def __call__(self, x_BxHxWxC: jax.Array) -> jax.Array:
        for layer in self.layers:
            x_BxHxWxC = layer(x_BxHxWxC)
            x_BxHxWxC = nnx.relu(x_BxHxWxC)
        return x_BxHxWxC


class VGG16(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.block_1 = ConvBlock(3, 64, 2, rngs)
        self.block_2 = ConvBlock(64, 128, 2, rngs)
        self.block_3 = ConvBlock(128, 256, 3, rngs)
        self.block_4 = ConvBlock(256, 512, 3, rngs)
        self.block_5 = ConvBlock(512, 512, 3, rngs)

    def __call__(self, x: jax.Array) -> LPIPSfeatures:
        mean = jnp.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3).astype(x.dtype)
        std = jnp.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3).astype(x.dtype)
        x = (x - mean) / std

        x = relu1_2 = self.block_1(x)
        x = max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = relu2_2 = self.block_2(x)
        x = max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = relu3_3 = self.block_3(x)
        x = max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = relu4_3 = self.block_4(x)
        x = max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = relu5_3 = self.block_5(x)

        features = LPIPSfeatures(
            relu1_2=relu1_2,
            relu2_2=relu2_2,
            relu3_3=relu3_3,
            relu4_3=relu4_3,
            relu5_3=relu5_3,
        )

        return features

    @classmethod
    def load_pretrained(cls) -> Self:
        weights_file = hf_hub_download(
            repo_id="lweitkamp/lpips-nnx", filename="vgg.pkl"
        )
        weights = pickle.load(open(weights_file, "rb"))

        graphdef, _ = nnx.split(cls(nnx.Rngs(0)))
        return nnx.merge(graphdef, weights)
