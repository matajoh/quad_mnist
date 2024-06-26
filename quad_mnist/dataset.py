from collections import deque
from itertools import permutations
import os
import pickle
from typing import List

import numpy as np
import requests
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm


def download(url: str, path: str):
    print(f"Downloading {url} to {path}")
    with open(path, 'wb') as file:
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=path) as progress:
            response = requests.get(url, stream=True)
            for chunk in response.iter_content(chunk_size=512):
                if chunk:
                    file.write(chunk)
                    progress.update(len(chunk))


def load_mnist():
    if not os.path.exists("mnist.pkl"):
        download("https://lectures.blob.core.windows.net/data/mnist.pkl", "mnist.pkl")

    with open("mnist.pkl", "rb") as mnist_pickle:
        return pickle.load(mnist_pickle, encoding="bytes")


def build_quadmnist():
    mnist = load_mnist()
    mnist_data = mnist["data"].reshape((-1, 1, 28, 28))
    mnist_target = mnist['target'].astype(np.int64)
    num_classes = 10
    num_train = 60000
    num_val = len(mnist_target) - num_train

    # group images by label
    labels = list(range(num_classes))
    train_images = {i: deque() for i in labels}
    val_images = {i: deque() for i in labels}
    for i, label in tqdm(enumerate(mnist_target), "Sorting images by label"):
        if i < num_train:
            train_images[label].append(i)
        else:
            val_images[label].append(i)

    print("Producing all quadrant permutations")
    combinations = np.stack(list(permutations(labels, 4))).astype(np.uint8)

    np.random.seed(20080524)

    # the training set is formed by all possible combinations
    data = []
    target = []
    num_batches = (num_train // len(combinations)) + 1
    for batch in range(num_batches):
        index = np.arange(len(combinations))
        np.random.shuffle(index)
        mixed = combinations[index]

        for combo in tqdm(mixed, f"Batch {batch}/{num_batches}: Training"):
            idx = [train_images[n].popleft() for n in combo]
            image = np.zeros((1, 56, 56), np.uint8)
            image[0, :28, :28] = mnist_data[idx[0]]
            image[0, :28, 28:] = mnist_data[idx[1]]
            image[0, 28:, :28] = mnist_data[idx[2]]
            image[0, 28:, 28:] = mnist_data[idx[3]]
            data.append(image)
            target.append(combo)
            for n, i in zip(combo, idx):
                train_images[n].append(i)

    # the validation set is formed of a randomized set of the remainder
    num_train = len(data)

    num_batches = (num_val // len(combinations)) + 1
    for batch in range(num_batches):
        index = np.arange(len(combinations))
        np.random.shuffle(index)
        mixed = combinations[index]

        for combo in tqdm(mixed, f"Batch {batch}/{num_batches}: Validation"):
            idx = [val_images[n].popleft() for n in combo]
            image = np.zeros((1, 56, 56), np.uint8)
            image[0, :28, :28] = mnist_data[idx[0]]
            image[0, :28, 28:] = mnist_data[idx[1]]
            image[0, 28:, :28] = mnist_data[idx[2]]
            image[0, 28:, 28:] = mnist_data[idx[3]]
            data.append(image)
            target.append(combo)
            for n, i in zip(combo, idx):
                val_images[n].append(i)

    data = np.stack(data)
    target = np.stack(target)
    np.savez("quad_mnist.npz", data=data, target=target, num_classes=num_classes, num_train=num_train, num_val=num_val)


def load_quadmnist():
    if not os.path.exists("quad_mnist.npz"):
        build_quadmnist()

    return np.load("quad_mnist.npz")


class MultilabelDataset:
    def __init__(self, train: TensorDataset, val: TensorDataset,
                 label_names: List[str] = None):
        if isinstance(train, TensorDataset):
            self.shape = train.tensors[0].shape[1:]
        else:
            self.shape = train.values.shape[1:]

        self.train = train
        self.val = val
        self.label_names = label_names
        self.mean = np.zeros((1,) + self.shape, np.float32)
        self.std = np.ones((1,) + self.shape, np.float32)

    def normalize(self) -> "MultilabelDataset":
        mean = self.train.values.mean(0, keepdims=True)
        std = self.train.values.std(0, keepdims=True)
        train = self.train.normalize(mean, std)
        val = self.val.normalize(mean, std)
        dataset = MultilabelDataset(train, val, self.label_names)
        dataset.mean = mean
        dataset.std = std
        return dataset

    def unnormalize(self, value: np.ndarray) -> np.ndarray:
        if len(value.shape) == 3:
            return (value.reshape((1,) + value.shape) * self.std + self.mean).reshape(value.shape)

        return value * self.std + self.mean

    def quadmnist() -> "MultilabelDataset":
        quadmnist = load_quadmnist()
        data = quadmnist["data"].astype(np.float32)  # convert the uint8s to floats
        data /= 255  # scale to be from 0 to 1
        target = quadmnist["target"].astype(np.int64)  # convert the uint8s to int32s
        num_classes = quadmnist["num_classes"].item()
        num_train = quadmnist["num_train"].item()

        def to_multihot(labels: np.ndarray) -> np.ndarray:
            num_targets = labels.shape[0]
            multihot = np.zeros((num_targets, num_classes), np.float32)
            for i in range(num_targets):
                multihot[i, labels[i]] = 1

            return multihot

        train = TensorDataset(torch.from_numpy(data[:num_train]), torch.from_numpy(to_multihot(target[:num_train])))
        val = TensorDataset(torch.from_numpy(data[num_train:]), torch.from_numpy(to_multihot(target[num_train:])))
        label_names = [str(i) for i in range(num_classes)]
        return MultilabelDataset(train, val, label_names)
