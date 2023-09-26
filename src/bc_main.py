import importlib
import pprint

import absl.app
import absl.flags
import jax
import jax.numpy as jnp
import numpy as np


from .data import (
    partition_batch_train_test,
    subsample_batch,
    preprocess_robot_dataset,
    augment_batch,
    get_data_augmentation,
    concatenate_batches,
)
from .jax_utils import JaxRNG, next_rng
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
    weight_decay_mask_tanh,
    weight_decay_mask_pretrain_tanh,
)
from .model import ResNetPolicy, TanhGaussianResNetPolicy, PretrainTanhGaussianResNetPolicy

from flax.training.train_state import TrainState
from functools import partial

FLAGS_DEF = define_flags_with_default(
    seed=42,
    dataset_path="",
    dataset_image_keys="side_image",
    image_augmentation="none",
    clip_action=0.99,
    train_ratio=0.9,
    batch_size=128,
    total_steps=10000,
    lr=1e-4,
    lr_warmup_steps=0,
    weight_decay=0.05,
    clip_gradient=1e9,
    log_freq=50,
    eval_freq=200,
    eval_batches=20,
    save_model=False,
    policy_class_name="TanhGaussianResNetPolicy",
    policy=TanhGaussianResNetPolicy.get_default_config(),
    logger=WandBLogger.get_default_config(),
)

FLAGS = absl.flags.FLAGS


def main(argv):
    assert FLAGS.dataset_path != ""
    policy_class = getattr(
        importlib.import_module("src.model"), FLAGS.policy_class_name
    )
    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logger, variant=variant)
    set_random_seed(FLAGS.seed)

    image_keys = FLAGS.dataset_image_keys.split(":")
    dataset = np.load(FLAGS.dataset_path, allow_pickle=True)
    if type(dataset) == np.ndarray and dataset.shape==():
        dataset = dataset.item()
    if type(dataset) == np.ndarray:
        dataset = concatenate_batches(dataset)
    elif type(dataset) == dict:
        pass
    else:
        raise TypeError
    dataset = preprocess_robot_dataset(dataset, FLAGS.clip_action)
    train_dataset, test_dataset = partition_batch_train_test(dataset, FLAGS.train_ratio)
    policy = policy_class(
        output_dim=dataset["action"].shape[-1],
        config_updates=FLAGS.policy,
    )
    params = policy.init(
        state=train_dataset["robot_state"][:5, ...],
        images=[dataset[key][:5, ...] for key in image_keys],
        rngs=next_rng(policy.rng_keys()),
    )

    learning_rate = get_learning_rate(FLAGS=FLAGS)

    if FLAGS.policy_class_name == "TanhGaussianResNetPolicy":
        optimizer = get_optimizer(
            FLAGS=FLAGS,
            learning_rate=learning_rate,
            weight_decay_mask=weight_decay_mask_tanh,
        )
    elif FLAGS.policy_class_name == "PretrainTanhGaussianResNetPolicy":
        optimizer = get_optimizer(
            FLAGS=FLAGS,
            learning_rate=learning_rate,
            weight_decay_mask=weight_decay_mask_pretrain_tanh,
        )
    else:
        raise ValueError(f"{FLAGS.policy_class_name} not Valid")

    train_state = TrainState.create(params=params, tx=optimizer, apply_fn=None)

    @partial(jax.jit, donate_argnums=1)
    def train_step(rng, train_state, state, action, images):
        rng_generator = JaxRNG(rng)

        def loss_fn(params):
            log_probs, mean = policy.apply(
                params,
                state,
                action,
                images,
                return_mean=True,
                method=policy.log_prob,
                rngs=rng_generator(policy.rng_keys()),
            )
            mse = jnp.mean(jnp.sum(jnp.square(mean - action), axis=-1))
            return -jnp.mean(log_probs), mse

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, mse), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            mse=mse,
            learning_rate=learning_rate(train_state.step),
        )
        return rng_generator(), train_state, metrics

    @jax.jit
    def eval_step(rng, train_state, state, action, images):
        rng_generator = JaxRNG(rng)
        log_probs, mean = policy.apply(
            train_state.params,
            state,
            action,
            images,
            return_mean=True,
            method=policy.log_prob,
            rngs=rng_generator(policy.rng_keys()),
        )
        loss = -jnp.mean(log_probs)
        mse = jnp.mean(jnp.sum(jnp.square(mean - action), axis=-1))
        metrics = dict(
            eval_loss=loss,
            eval_mse=mse,
        )
        return rng_generator(), metrics

    augmentation = get_data_augmentation(FLAGS.image_augmentation)
    rng = next_rng()

    best_loss, best_mse = float("inf"), float("inf")
    best_loss_model, best_mse_model = None, None

    for step in range(FLAGS.total_steps):
        batch = subsample_batch(train_dataset, FLAGS.batch_size)
        batch = augment_batch(augmentation, batch)
        rng, train_state, metrics = train_step(
            rng,
            train_state,
            batch["robot_state"],
            batch["action"],
            [batch[key] for key in image_keys],
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
                    batch["action"],
                    [batch[key] for key in image_keys],
                )
                eval_metrics.append(metrics)
            eval_metrics = average_metrics(jax.device_get(eval_metrics))
            eval_metrics["step"] = step

            if eval_metrics["eval_loss"] < best_loss:
                best_loss = eval_metrics["eval_loss"]
                best_loss_model = jax.device_get(train_state)

            if eval_metrics["eval_mse"] < best_mse:
                best_mse = eval_metrics["eval_mse"]
                best_mse_model = jax.device_get(train_state)

            eval_metrics["best_loss"] = best_loss
            eval_metrics["best_mse"] = best_mse
            wandb_logger.log(eval_metrics)
            pprint.pprint(eval_metrics)

            if FLAGS.save_model:
                save_data = {
                    "variant": variant,
                    "step": step,
                    "train_state": jax.device_get(train_state),
                    "best_loss_model": best_loss_model,
                    "best_mse_model": best_mse_model,
                }
                wandb_logger.save_pickle(save_data, f"model.pkl")

    if FLAGS.save_model:
        save_data = {
            "variant": variant,
            "step": step,
            "train_state": jax.device_get(train_state),
            "best_loss_model": best_loss_model,
            "best_mse_model": best_mse_model,
        }
        wandb_logger.save_pickle(save_data, f"model.pkl")


if __name__ == "__main__":
    absl.app.run(main)
