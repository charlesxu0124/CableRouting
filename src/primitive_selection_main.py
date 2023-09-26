import pprint

import absl.app
import absl.flags
import cloudpickle as pickle
import jax
import jax.numpy as jnp
import numpy as np
import optax

from functools import partial

from flax.training.train_state import TrainState

from .data import (
    partition_batch_train_test,
    subsample_batch,
    preprocess_robot_dataset,
    augment_batch,
    get_data_augmentation,
    concatenate_batches,
)

from .jax_utils import (
    JaxRNG,
    next_rng,
    next_rng,
)

from .model import PrimitiveSelectionPolicy, PretrainTanhGaussianResNetPolicy

from .utils import (
    define_flags_with_default,
    set_random_seed,
    get_user_flags,
    WandBLogger,
    average_metrics,
)

from .train_utils import (
    get_learning_rate,
    get_optimizer,
    weight_decay_mask_primitive,
)

FLAGS_DEF = define_flags_with_default(
    seed=42,
    dataset_path="",
    dataset_image_keys="side_image",
    image_augmentation="none",
    clip_action=0.99,
    train_ratio=0.9,
    batch_size=128,
    total_steps=10000,
    finetune_steps=500,
    lr=1e-4,
    lr_warmup_steps=0,
    weight_decay=0.05,
    clip_gradient=1e9,
    log_freq=50,
    eval_freq=200,
    eval_batches=20,
    save_model=False,
    policy=PrimitiveSelectionPolicy.get_default_config(),
    logger=WandBLogger.get_default_config(),
    gripper=False,
    encoder_checkpoint_path="",  # resnet features
    primitive_policy_checkpoint_path="",  # primitive selection policy
    pretrained_model_key="train_state",
    output_dim_gaussian_policy=0,
    finetune_policy=False,
)


FLAGS = absl.flags.FLAGS


def load_policy_and_parms(ckpt_path, policy_config, model_key):
    assert ckpt_path != ""
    with open(ckpt_path, "rb") as fin:
        checkpoint_data = pickle.load(fin)
    checkpoint_policy_config = {
        k[7:]: v
        for k, v in checkpoint_data["variant"].items()
        if k.startswith("policy.")
    }
    policy_config.update_from_flattened_dict(checkpoint_policy_config)
    params = jax.device_put(checkpoint_data[model_key].params)
    return policy_config, params


def main(argv):
    assert FLAGS.dataset_path != ""
    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logger, variant=variant)
    set_random_seed(FLAGS.seed)

    image_keys = FLAGS.dataset_image_keys.split(":")
    dataset_paths = FLAGS.dataset_path.split(":")
    dataset = [
        np.load(dataset_path, allow_pickle=True).item()
        for dataset_path in dataset_paths
    ]
    dataset = concatenate_batches(dataset)
    dataset = preprocess_robot_dataset(dataset, FLAGS.clip_action)
    train_dataset, test_dataset = partition_batch_train_test(dataset, FLAGS.train_ratio)
    pretrain_features_policy_config = PretrainTanhGaussianResNetPolicy.get_default_config()
    pretrain_features_policy_config, pretrain_features_policy_params = load_policy_and_parms(
        FLAGS.encoder_checkpoint_path,
        pretrain_features_policy_config,
        FLAGS.pretrained_model_key,
    )
    pretrain_features_policy = PretrainTanhGaussianResNetPolicy(
        output_dim=4, 
        config_updates=pretrain_features_policy_config
    )
    print("State: ", pretrain_features_policy_config.state_injection)

    policy = PrimitiveSelectionPolicy(
        output_dim_gaussian_policy=FLAGS.output_dim_gaussian_policy,
        config_updates=FLAGS.policy,
    )
    rng = next_rng()

    def forward_pretrained_policy(rng, robot_state, images):
        rng_generator = JaxRNG(rng)
        features, _, _ = pretrain_features_policy.apply(
            pretrain_features_policy_params,
            robot_state,
            images,
            deterministic=False,
            rngs=rng_generator(policy.rng_keys()),
            return_features=True,
        )

        return jax.lax.stop_gradient(features)

    features_init = forward_pretrained_policy(
        rng,
        robot_state=train_dataset["robot_state"][:5, ...],
        images=[dataset[key][:5, ...] for key in image_keys],
    )
    if not FLAGS.finetune_policy:
        params = policy.init(
            features=features_init,
            primitive_sequence=jnp.zeros((5, 6), dtype=jnp.int32),
            rngs=next_rng(policy.rng_keys()),
        )
    else:
        _, params = load_policy_and_parms(
            FLAGS.primitive_policy_checkpoint_path,
            FLAGS.policy,
            FLAGS.pretrained_model_key,
        )
    learning_rate = get_learning_rate(FLAGS=FLAGS)

    optimizer = get_optimizer(FLAGS, learning_rate, weight_decay_mask_primitive)
    train_state = TrainState.create(params=params, tx=optimizer, apply_fn=None)

    @partial(jax.jit, donate_argnums=1)
    def train_step(rng, train_state, state, images, primitive_sequence, labels):
        rng_generator = JaxRNG(rng)
        features = forward_pretrained_policy(rng_generator(), state, images)

        def loss_fn(params):
            logits = policy.apply(
                params,
                features,
                primitive_sequence,
                rngs=rng_generator(policy.rng_keys()),
            )
            cross_entropy = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits, labels)
            )
            accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
            return cross_entropy, accuracy

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, accuracy), grads = grad_fn(train_state.params)

        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            train_accuracy=accuracy,
            learning_rate=learning_rate(train_state.step),
        )
        return rng_generator(), train_state, metrics

    @jax.jit
    def eval_step(rng, train_state, state, images, primitive_sequence, labels):
        rng_generator = JaxRNG(rng)
        features = forward_pretrained_policy(rng_generator(), state, images)
        logits = policy.apply(
            train_state.params,
            features,
            primitive_sequence,
            rngs=rng_generator(policy.rng_keys()),
        )
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
        metrics = dict(
            eval_loss=loss,
            eval_accuracy=accuracy,
        )
        return rng_generator(), metrics

    augmentation = get_data_augmentation(FLAGS.image_augmentation)
    rng = next_rng()

    best_loss = float("inf")
    best_loss_model = None
    if FLAGS.finetune_policy:
        train_steps = FLAGS.finetune_steps
    else:
        train_steps = FLAGS.total_steps

    for step in range(train_steps):
        batch = subsample_batch(train_dataset, FLAGS.batch_size)
        batch = augment_batch(augmentation, batch)
        rng, train_state, metrics = train_step(
            rng,
            train_state,
            batch["robot_state"],
            [batch[key] for key in image_keys],
            batch["primitive_sequence"],
            batch["labels"],
        )
        metrics["step"] = step

        if step % FLAGS.log_freq == 0:
            wandb_logger.log(metrics)
            pprint.pprint(metrics)

        if step % FLAGS.eval_freq == 0:
            eval_metrics = []
            for _ in range(FLAGS.eval_batches):
                batch = subsample_batch(test_dataset, FLAGS.batch_size)
                rng, metrics = eval_step(
                    rng,
                    train_state,
                    batch["robot_state"],
                    [batch[key] for key in image_keys],
                    batch["primitive_sequence"],
                    batch["labels"],
                )
                eval_metrics.append(metrics)
            eval_metrics = average_metrics(jax.device_get(eval_metrics))
            eval_metrics["step"] = step

            if eval_metrics["eval_loss"] < best_loss:
                best_loss = eval_metrics["eval_loss"]
                best_loss_model = jax.device_get(train_state)

            eval_metrics["best_loss"] = best_loss
            wandb_logger.log(eval_metrics)
            pprint.pprint(eval_metrics)

        if step % 3000 == 0:
            if FLAGS.save_model:
                save_data = {
                    "variant": variant,
                    "step": step,
                    "train_state": jax.device_get(train_state),
                    "best_loss_model": best_loss_model,
                }
                wandb_logger.save_pickle(save_data, f"model_{step}_steps.pkl")
        if FLAGS.finetune_policy and step % 100 == 0:
            if FLAGS.save_model:
                save_data = {
                    "variant": variant,
                    "step": step,
                    "train_state": jax.device_get(train_state),
                    "best_loss_model": best_loss_model,
                }
                wandb_logger.save_pickle(save_data, f"model_{step}_steps.pkl")


if __name__ == "__main__":
    absl.app.run(main)
