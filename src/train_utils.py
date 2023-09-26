import re

import optax


from .jax_utils import named_tree_map


from .model import (
    PrimitiveSelectionPolicy,
    PretrainTanhGaussianResNetPolicy,
    TanhGaussianResNetPolicy,
)


def get_learning_rate(FLAGS, init=0.0, end=0.0):
    return optax.warmup_cosine_decay_schedule(
        init_value=init,
        peak_value=FLAGS.lr,
        warmup_steps=FLAGS.lr_warmup_steps,
        decay_steps=FLAGS.total_steps,
        end_value=end,
    )


def weight_decay_mask_primitive(params):
    def decay(name, _):
        for rule in PrimitiveSelectionPolicy.get_weight_decay_exclusions():
            if re.search(rule, name) is not None:
                return False
        return True

    return named_tree_map(decay, params, sep="/")


def weight_decay_mask_pretrain_tanh(params):
    def decay(name, _):
        for rule in PretrainTanhGaussianResNetPolicy.get_weight_decay_exclusions():
            if re.search(rule, name) is not None:
                return False
        return True

    return named_tree_map(decay, params, sep="/")


def weight_decay_mask_tanh(params):
    def decay(name, _):
        for rule in TanhGaussianResNetPolicy.get_weight_decay_exclusions():
            if re.search(rule, name) is not None:
                return False
        return True

    return named_tree_map(decay, params, sep="/")


def get_optimizer(FLAGS, learning_rate, weight_decay_mask):
    return optax.chain(
        optax.clip_by_global_norm(FLAGS.clip_gradient),
        optax.adamw(
            learning_rate=learning_rate,
            weight_decay=FLAGS.weight_decay,
            mask=weight_decay_mask,
        ),
    )
