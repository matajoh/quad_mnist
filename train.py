import os
from typing import List, NamedTuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import quad_mnist as qm
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def compute_accuracy(output, expected_labels):
    actual_labels = torch.where(output.sigmoid() > 0.5, 1, 0)
    num_correct = (actual_labels == expected_labels).sum().item()
    total = expected_labels.shape[0] * expected_labels.shape[1]
    return num_correct / total


Snapshot = NamedTuple("Snapshot", [("step", int), ("accuracy", float), ("loss", float)])


def train_model(dataset: qm.MultilabelDataset, net: nn.Module, criterion, batch_size=100, num_epochs=20, device="cpu") -> List[Snapshot]:
    """
    Trains a PyTorch model.

    Arguments:

    model -- The model to train

    Keyword arguments:

    batchsize -- The batchsize to use during training
    epoch -- The number of epochs to train
    """

    net.to(device)
    optimizer = optim.Adam(net.parameters())

    snapshots = []
    train_loader = torch.utils.data.DataLoader(dataset.train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset.val, batch_size=batch_size, shuffle=False)
    num_train = len(dataset.train)
    for epoch in range(1, num_epochs + 1):
        print('epoch', epoch)

        # training
        net.train()
        sum_accuracy = 0
        sum_loss = 0
        total = 0

        with tqdm(total=num_train, desc=f'Epoch {epoch}/{num_epochs}', unit='img') as pbar:
            for x, t in train_loader:
                x = x.to(device)
                t = t.to(device)

                optimizer.zero_grad()
                y = net(x)
                loss = criterion(y, t)
                accuracy = compute_accuracy(y, t)
                loss.backward()
                optimizer.step()

                sum_loss += loss.item() * len(t)
                sum_accuracy += accuracy * len(t)
                total += len(t)
                
                pbar.update(x.shape[0])
                pbar.set_postfix(**{"loss (batch)": loss.item()})

        # evaluation
        net.eval()
        sum_accuracy = 0
        sum_loss = 0
        total = 0
        for x, t in tqdm(val_loader, desc="Validation", unit="batch"):
            x = x.to(device)
            t = t.to(device)
            y = net(x)
            loss = criterion(y, t)
            accuracy = compute_accuracy(y, t)

            sum_loss += loss.item() * len(t)
            sum_accuracy += accuracy * len(t)
            total += len(t)

        print('validation  mean loss={}, accuracy={}'.format(sum_loss / total, sum_accuracy / total))
        snapshots.append(Snapshot(epoch, sum_accuracy / total, sum_loss / total))
    
    return snapshots


def evaluate_model(dataset: qm.MultilabelDataset, net: nn.Module, snapshots: List[Snapshot], device="cpu"):
    label_names = dataset.label_names
    num_examples = 6
    examples, _ = dataset.val[0:num_examples]
    columns = num_examples // 2

    net.to(device)
    net.eval()
    x = examples.to(device)
    output = F.sigmoid(net(x)).cpu().detach().numpy()
    top_labels = np.argpartition(output, -4)[:, -4:]

    plt.rc('font', size=15)
    fig = plt.figure(figsize=(10, 4))
    gs0 = gridspec.GridSpec(1, 2, wspace=.3)
    gs00 = gridspec.GridSpecFromSubplotSpec(2, columns, gs0[0])

    for i in range(num_examples):
        row = i // columns
        col = i % columns
        ax = fig.add_subplot(gs00[row, col])
        if examples[i].shape[0] == 3:
            reorder = dataset.unnormalize(examples[i].detach().cpu().numpy())
            reorder = np.swapaxes(reorder, 0, 1)
            reorder = np.swapaxes(reorder, 1, 2)
            ax.imshow(reorder, interpolation='nearest')
        else:
            ax.imshow(examples[i].reshape(examples[i].shape[1:]), cmap='gray', interpolation='nearest')

        text3 = label_names[top_labels[i, 0]] if output[i, top_labels[i, 0]] > 0.5 else ""
        text2 = label_names[top_labels[i, 1]] if output[i, top_labels[i, 1]] > 0.5 else ""
        text1 = label_names[top_labels[i, 2]] if output[i, top_labels[i, 2]] > 0.5 else ""
        text0 = label_names[top_labels[i, 3]] if output[i, top_labels[i, 3]] > 0.5 else ""
        ax.set_xlabel("{} {}\n{} {}".format(text0, text1, text2, text3))
        ax.set_xticks([])
        ax.set_yticks([])

    ax1 = fig.add_subplot(gs0[1])
    x = [snapshots[i][0] for i in range(len(snapshots))]
    y1 = [snapshots[i][1] for i in range(len(snapshots))]
    y2 = [snapshots[i][2] for i in range(len(snapshots))]

    acc_line = ax1.plot(x, y1, 'b-')
    ax1.set_xticks([])
    ax1.set_ylim(0, 1.1)
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()

    loss_line = ax2.plot(x, y2, 'g--')
    ax2.tick_params('y', colors='g')

    ax2.legend(acc_line + loss_line, ["Accuracy", "Loss"], loc="center right")


def main():
    dataset = qm.MultilabelDataset.quadmnist()
    if dataset.label_frequency is not None:
        weight = np.exp(-1.0 / dataset.label_frequency.astype(np.float32))
        weight /= weight.sum()
        weight = torch.from_numpy(weight)
    else:
        weight = None

    net = qm.FCNMulti()
    path = "quad_mnist.results"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss(weight.to(device))

    if os.path.exists(path):
        results = torch.load(path)
        snapshots = results["snapshots"]
        print(path, "loaded")
    else:
        print("Running on", device)
        snapshots = train_model(dataset, net, criterion, device=device)
        results = {"net": net.state_dict(), "snapshots": snapshots}
        torch.save(results, path)

    net.load_state_dict(results["net"])
    evaluate_model(dataset, net, snapshots, device=device)
    plt.savefig("quad_mnist.png")


if __name__ == "__main__":
    main()
