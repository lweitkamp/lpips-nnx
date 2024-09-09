import jax
import jax.numpy as jnp

from lpips_nnx.vgg import VGG16, LPIPSfeatures
from lpips_nnx.linear_weights import LPIPSWeight


def normalize(x, eps=1e-10):
    return x * jax.lax.rsqrt((x * x).sum(axis=-1, keepdims=True) + eps)


class LPIPS:
    def __init__(self):
        self.vgg: VGG16 = VGG16.load_pretrained()
        self.weights: LPIPSWeight = LPIPSWeight.load_pretrained()

    def __call__(self, x_BxHxWxC: jax.Array, x0_BxHxWxC: jax.Array):
        if x_BxHxWxC.ndim == 3:
            x_BxHxWxC = jnp.expand_dims(x_BxHxWxC, axis=0)
        
        if x0_BxHxWxC.ndim == 3:
            x0_BxHxWxC = jnp.expand_dims(x0_BxHxWxC, axis=0)

        f_features: LPIPSfeatures = self.vgg((x_BxHxWxC + 1) / 2)
        f0_features: LPIPSfeatures = self.vgg((x0_BxHxWxC + 1) / 2)

        res_Bx1 = jnp.zeros((x_BxHxWxC.shape[0], 1))
        for i, feature in enumerate(LPIPSfeatures._fields):
            d_WxHxD = (normalize(f_features[i]) - normalize(f0_features[i])) ** 2
            d_WxHx1 = self.weights(d_WxHxD, feature)
            res_Bx1 += d_WxHx1.mean(axis=(1, 2))

        return res_Bx1
