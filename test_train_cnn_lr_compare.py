import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import os

np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, 28, 28)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

idx = np.random.permutation(np.arange(num))

with open('idx_lr_compare.pickle', 'wb') as f:
    pickle.dump(idx, f)

train_imgs = train_imgs[idx]
train_labs = train_labs[idx]

valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]

train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

train_imgs = train_imgs.astype(np.float32) / 255.0
valid_imgs = valid_imgs.astype(np.float32) / 255.0

learning_rates = [0.001, 0.01, 0.05]

for lr in learning_rates:
    print("=" * 60)
    print(f"Training CNN with learning rate = {lr}")
    print("=" * 60)

    cnn_model = nn.models.Model_CNN()

    optimizer = nn.optimizer.SGD(
        init_lr=lr,
        model=cnn_model
    )

    loss_fn = nn.op.MultiCrossEntropyLoss(
        model=cnn_model,
        max_classes=train_labs.max() + 1
    )

    runner = nn.runner.RunnerM(
        cnn_model,
        optimizer,
        nn.metric.accuracy,
        loss_fn,
        scheduler=None
    )

    lr_name = str(lr).replace('.', 'p')
    save_dir = f'./best_models_cnn_lr_{lr_name}'

    runner.train(
        [train_imgs, train_labs],
        [valid_imgs, valid_labs],
        num_epochs=5,
        log_iters=100,
        save_dir=save_dir
    )

    fig, axes = plt.subplots(1, 2)
    axes.reshape(-1)
    fig.set_tight_layout(1)
    plot(runner, axes)

    curve_path = f'./cnn_lr_{lr_name}_curve.png'
    plt.savefig(curve_path, dpi=300)
    plt.close()

    print(f"Finished learning rate = {lr}")
    print(f"Model saved in: {save_dir}")
    print(f"Learning curve saved as: {curve_path}")