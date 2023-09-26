import numpy as np
import torch

from copy import deepcopy
from torchvision.transforms import RandAugment, TrivialAugmentWide, AugMix


def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices, ...]
    return indexed


def partition_batch_train_test(batch, train_ratio, random=False):
    length = batch[list(batch.keys())[0]].shape[0]
    if random:
        train_indices = np.random.rand(length) < train_ratio
    else:
        train_indices = np.linspace(0, 1, length) < train_ratio
    train_batch = index_batch(batch, train_indices)
    test_batch = index_batch(batch, ~train_indices)
    return train_batch, test_batch


def subsample_batch(batch, size):
    length = batch[list(batch.keys())[0]].shape[0]
    indices = np.random.randint(length, size=size)
    return index_batch(batch, indices)


def concatenate_batches(batches):
    concatenated = {}
    for key in batches[0].keys():
        concatenated[key] = np.concatenate([batch[key] for batch in batches], axis=0)
    return concatenated


def split_batch(batch, batch_size):
    batches = []
    length = batch[list(batch.keys())[0]].shape[0]
    keys = batch.keys()
    for start in range(0, length, batch_size):
        end = min(start + batch_size, length)
        batches.append({key: batch[key][start:end, ...] for key in keys})
    return batches


def preprocess_robot_dataset(dataset, clip_action):
    dataset = deepcopy(dataset)
    for key in ("side_image", "wrist45_image", "wrist225_image"):
        dataset[key] = dataset[key].astype(np.float32) / 255.0
    if "action" in dataset.keys():
        dataset["action"] = np.clip(
            dataset["action"], -clip_action, clip_action
        ).astype(np.float32)
    dataset["robot_state"] = dataset["robot_state"].astype(np.float32)
    return dataset


def get_data_augmentation(augmentation):
    if augmentation == "none":
        return None
    elif augmentation == "rand":
        return torch.jit.script(RandAugment())
    elif augmentation == "trivial":
        return torch.jit.script(TrivialAugmentWide())
    elif augmentation == "augmix":
        return torch.jit.script(AugMix())
    else:
        raise ValueError("Unsupported augmentation type!")


def augment_images(augmentation, image):
    if augmentation is None:
        return image

    # Switch to channel first
    image = np.clip(image, 0.0, 1.0)
    image = np.transpose((image * 255.0).astype(np.ubyte), (0, 3, 1, 2))
    with torch.no_grad():
        image = torch.from_numpy(image)
        image = augmentation(image)
        image = image.numpy()

    # Switch to channel last
    image = np.transpose(image, (0, 2, 3, 1)).astype(np.float32) / 255.0
    return image


def augment_batch(augmentation, batch):
    batch = deepcopy(batch)
    for key in ("side_image", "wrist45_image", "wrist225_image"):
        batch[key] = augment_images(augmentation, batch[key])
    return batch
