import distrax
import jax.numpy as jnp

from flax import linen as nn
from functools import partial
from ml_collections import ConfigDict
from typing import Any, Callable, Sequence, Tuple


class Scalar(nn.Module):
    init_value: float

    def setup(self):
        self.value = self.param("value", lambda x: self.init_value)

    def __call__(self):
        return self.value


class FullyConnectedNetwork(nn.Module):
    output_dim: int
    arch: str = "256-256"

    @nn.compact
    def __call__(self, input_tensor):
        x = input_tensor
        hidden_sizes = [int(h) for h in self.arch.split("-")]
        for h in hidden_sizes:
            x = nn.Dense(h)(x)
            x = nn.relu(x)

        return nn.Dense(self.output_dim)(x)


class ResNetBlock(nn.Module):
    """ResNet block."""

    filters: int
    conv: Any
    norm: Any
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(
                residual
            )
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""

    filters: int
    conv: Any
    norm: Any
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(
                self.filters * 4, (1, 1), self.strides, name="conv_proj"
            )(residual)
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class ResNet(nn.Module):
    """ResNetV1."""

    stage_sizes: Sequence[int]
    block_cls: Any
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    conv: Any = nn.Conv

    @nn.compact
    def __call__(self, x):
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.GroupNorm,
            num_groups=32,
            dtype=self.dtype,
        )

        x = conv(
            self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], name="conv_init"
        )(x)
        x = norm(name="bn_init")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2**i,
                    strides=strides,
                    conv=conv,
                    norm=norm,
                    act=self.act,
                )(x)
        x = jnp.asarray(x, self.dtype)
        return x


ResNetModules = {
    "ResNet18": partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock),
    "ResNet34": partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock),
    "ResNet50": partial(
        ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock
    ),
    "ResNet101": partial(
        ResNet, stage_sizes=[3, 4, 23, 3], block_cls=BottleneckResNetBlock
    ),
    "ResNet152": partial(
        ResNet, stage_sizes=[3, 8, 36, 3], block_cls=BottleneckResNetBlock
    ),
    "ResNet200": partial(
        ResNet, stage_sizes=[3, 24, 36, 3], block_cls=BottleneckResNetBlock
    ),
}


class ResNetPolicy(nn.Module):
    output_dim: int
    config_updates: ... = None

    @staticmethod
    @nn.nowrap
    def get_default_config(updates=None):
        config = ConfigDict()
        config.resnet_type = "ResNet18"
        config.spatial_aggregate = "average"
        config.mlp_arch = "256-256"
        config.state_injection = "full"
        config.state_projection_dim = 64
        config.share_resnet_between_views = True

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        return config

    @staticmethod
    @nn.nowrap
    def rng_keys():
        return ("params",)

    @staticmethod
    @nn.nowrap
    def get_weight_decay_exclusions():
        return "bias"

    def setup(self):
        self.config = self.get_default_config(self.config_updates)

    @nn.compact
    def __call__(self, state, images, return_features=False):
        features = []
        if self.config.share_resnet_between_views:
            resnet = ResNetModules[self.config.resnet_type]()
        for x in images:
            if self.config.share_resnet_between_views:
                z = resnet(x)
            else:
                z = ResNetModules[self.config.resnet_type]()(x)

            if self.config.spatial_aggregate == "average":
                z = jnp.mean(z, axis=(1, 2))
            elif self.config.spatial_aggregate == "flatten":
                z = z.reshape(z.shape[0], -1)
            else:
                raise ValueError("Unsupported spatial aggregation type!")
            features.append(z)

        if self.config.state_injection == "full":
            features.append(nn.Dense(self.config.state_projection_dim)(state))
        elif self.config.state_injection == "z_only":
            features.append(nn.Dense(self.config.state_projection_dim)(state[:, 2:3]))
        elif self.config.state_injection == "none":
            pass
        else:
            raise ValueError(
                f"Unsupported state_injection: {self.config.state_injection}!"
            )

        features = jnp.concatenate(features, axis=1)
        fc_out = FullyConnectedNetwork(self.output_dim, self.config.mlp_arch)(features)
        if return_features:
            return features, fc_out
        else:
            return fc_out


class TanhGaussianResNetPolicy(nn.Module):
    output_dim: int
    config_updates: ... = None

    @staticmethod
    @nn.nowrap
    def get_default_config(updates=None):
        return ResNetPolicy.get_default_config(updates)

    @staticmethod
    @nn.nowrap
    def rng_keys():
        return ("params", "noise")

    @staticmethod
    @nn.nowrap
    def get_weight_decay_exclusions():
        return "bias"

    def setup(self):
        self.config = self.get_default_config(self.config_updates)
        self.backbone = ResNetPolicy(self.output_dim * 2, self.config)

    def log_prob(self, state, action, images, return_mean=False):
        mean, log_std = jnp.split(self.backbone(state, images), 2, axis=-1)
        log_std = jnp.clip(log_std, -20.0, 2.0)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
            distrax.Block(distrax.Tanh(), ndims=1),
        )
        log_probs = action_distribution.log_prob(action)
        if return_mean:
            return log_probs, mean
        return log_probs

    def __call__(self, state, images, deterministic=False):
        mean, log_std = jnp.split(self.backbone(state, images), 2, axis=-1)
        log_std = jnp.clip(log_std, -20.0, 2.0)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
            distrax.Block(distrax.Tanh(), ndims=1),
        )
        if deterministic:
            samples = jnp.tanh(mean)
            log_prob = action_distribution.log_prob(samples)
        else:
            samples, log_prob = action_distribution.sample_and_log_prob(
                seed=self.make_rng("noise")
            )

        return samples, log_prob


class PretrainTanhGaussianResNetPolicy(nn.Module):
    output_dim: int
    config_updates: ... = None

    @staticmethod
    @nn.nowrap
    def get_default_config(updates=None):
        return ResNetPolicy.get_default_config(updates)

    @staticmethod
    @nn.nowrap
    def rng_keys():
        return ("params", "noise")

    @staticmethod
    @nn.nowrap
    def get_weight_decay_exclusions():
        return "bias"

    def setup(self):
        self.config = self.get_default_config(self.config_updates)
        self.backbone = ResNetPolicy(self.output_dim * 2, self.config)

    def log_prob(self, state, action, images, return_mean=False):
        mean, log_std = jnp.split(
            self.backbone(state, images, return_features=False), 2, axis=-1
        )

        log_std = jnp.clip(log_std, -20.0, 2.0)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
            distrax.Block(distrax.Tanh(), ndims=1),
        )
        log_probs = action_distribution.log_prob(action)
        if return_mean:
            return log_probs, mean
        return log_probs

    def __call__(self, state, images, deterministic=False, return_features=False):
        features, fc_out = self.backbone(state, images, return_features=True)
        mean, log_std = jnp.split(fc_out, 2, axis=-1)
        log_std = jnp.clip(log_std, -20.0, 2.0)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
            distrax.Block(distrax.Tanh(), ndims=1),
        )
        if deterministic:
            samples = jnp.tanh(mean)
            log_prob = action_distribution.log_prob(samples)
        else:
            samples, log_prob = action_distribution.sample_and_log_prob(
                seed=self.make_rng("noise")
            )
        if return_features:
            return features, samples, log_prob
        else:
            return samples, log_prob


class PrimitiveSelectionPolicy(nn.Module):
    output_dim_gaussian_policy: int
    config_updates: ... = None
    mlp_arch: str = "256-256"
    total_num_primitives: int = 4
    num_embeddings: int = 5  # 0 is padding, 1-3 are actual primitives, 4 is go next
    num_embedding_features: int = 4
    primitive_sequence_legnth: int = 6

    @staticmethod
    @nn.nowrap
    def get_default_config(updates=None):
        return ResNetPolicy.get_default_config(updates)

    @staticmethod
    @nn.nowrap
    def rng_keys():
        return ("params", "noise")

    @staticmethod
    @nn.nowrap
    def get_weight_decay_exclusions():
        return "bias"

    def setup(self):
        self.config = self.get_default_config(self.config_updates)
        self.embed = nn.Embed(
            self.num_embeddings,
            self.num_embedding_features,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )

    @nn.compact
    def __call__(self, features, primitive_sequence, deterministic=False):
        assert primitive_sequence.shape[-1] == self.primitive_sequence_legnth
        features = nn.Dense(256)(features)
        features = nn.relu(features)
        primitives_embeddings = self.embed(primitive_sequence)
        primitives_embeddings = primitives_embeddings.reshape(
            (primitives_embeddings.shape[0], -1)
        )
        primitives_embeddings = nn.Dense(256)(primitives_embeddings)
        primitives_embeddings = nn.relu(primitives_embeddings)
        embedding = jnp.concatenate([features, primitives_embeddings], axis=-1)
        layer1 = nn.Dense(256)(embedding)
        logits = nn.Dense(self.total_num_primitives)(layer1)
        return logits
