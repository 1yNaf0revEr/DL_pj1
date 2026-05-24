import os
import glob
import gzip
import numpy as np
import matplotlib.pyplot as plt
from struct import unpack

import mynn as nn


OUTPUT_DIR = "./direction5_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_latest_model(folder):
    files = glob.glob(os.path.join(folder, "*.pickle"))
    if len(files) == 0:
        raise FileNotFoundError(f"No .pickle model file found in {folder}")
    return max(files, key=os.path.getmtime)


def load_mnist_test_for_mlp():
    test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
    test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    test_imgs = test_imgs.astype(np.float32) / 255.0
    return test_imgs, test_labs


def load_mnist_test_for_cnn():
    test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
    test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, 28, 28)

    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    test_imgs = test_imgs.astype(np.float32) / 255.0
    return test_imgs, test_labs


def predict_in_batches(model, X, batch_size=256):
    logits_list = []

    for start in range(0, len(X), batch_size):
        end = start + batch_size
        logits = model(X[start:end])
        logits_list.append(logits)

    return np.concatenate(logits_list, axis=0)


def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    return cm


def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    classes = np.arange(10)
    plt.xticks(classes)
    plt.yticks(classes)

    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_misclassified_examples(images, y_true, y_pred, model_name, save_path, is_cnn=False, max_num=16):
    wrong_idx = np.where(y_true != y_pred)[0]

    if len(wrong_idx) == 0:
        print(f"{model_name}: no misclassified samples found.")
        return

    selected = wrong_idx[:max_num]

    plt.figure(figsize=(8, 8))

    for i, idx in enumerate(selected):
        plt.subplot(4, 4, i + 1)

        if is_cnn:
            img = images[idx, 0]
        else:
            img = images[idx].reshape(28, 28)

        plt.imshow(img, cmap="gray")
        plt.title(f"T:{y_true[idx]} P:{y_pred[idx]}")
        plt.axis("off")

    plt.suptitle(f"{model_name} Misclassified Examples")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_mlp_first_layer_weights(model, save_path, max_num=16):
    first_linear = None

    for layer in model.layers:
        if layer.optimizable:
            first_linear = layer
            break

    if first_linear is None:
        print("No linear layer found in MLP.")
        return

    W = first_linear.W

    plt.figure(figsize=(8, 8))

    for i in range(max_num):
        weight_img = W[:, i].reshape(28, 28)

        plt.subplot(4, 4, i + 1)
        plt.imshow(weight_img)
        plt.title(f"H{i}")
        plt.axis("off")

    plt.suptitle("MLP First-layer Weight Visualization")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_cnn_conv_kernels(model, save_path):
    W = model.conv1.W
    out_channels = W.shape[0]

    plt.figure(figsize=(10, 3))

    for i in range(out_channels):
        kernel = W[i, 0]

        plt.subplot(1, out_channels, i + 1)
        plt.imshow(kernel)
        plt.title(f"K{i}")
        plt.axis("off")

    plt.suptitle("CNN First Convolution Kernels")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def print_error_summary(model_name, cm):
    total = np.sum(cm)
    correct = np.trace(cm)
    acc = correct / total

    print("=" * 60)
    print(model_name)
    print("=" * 60)
    print(f"Test accuracy: {acc:.4f}")

    print("\nPer-class accuracy:")
    for i in range(10):
        class_total = np.sum(cm[i])
        class_correct = cm[i, i]
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"Digit {i}: {class_acc:.4f}")

    pairs = []

    for i in range(10):
        for j in range(10):
            if i != j:
                pairs.append((cm[i, j], i, j))

    pairs.sort(reverse=True)

    print("\nTop confused pairs:")
    for count, true_label, pred_label in pairs[:10]:
        if count > 0:
            print(f"True {true_label} predicted as {pred_label}: {count}")

    print()


def analyze_mlp():
    mlp_model_path = find_latest_model("./best_models")
    print("Loading MLP model from:", mlp_model_path)

    model = nn.models.Model_MLP()
    model.load_model(mlp_model_path)

    test_imgs, test_labs = load_mnist_test_for_mlp()

    logits = predict_in_batches(model, test_imgs, batch_size=512)
    preds = np.argmax(logits, axis=1)

    cm = compute_confusion_matrix(test_labs, preds)

    plot_confusion_matrix(
        cm,
        "MLP Confusion Matrix on MNIST",
        os.path.join(OUTPUT_DIR, "mlp_confusion_matrix.png")
    )

    plot_misclassified_examples(
        test_imgs,
        test_labs,
        preds,
        "MLP",
        os.path.join(OUTPUT_DIR, "mlp_misclassified_examples.png"),
        is_cnn=False
    )

    plot_mlp_first_layer_weights(
        model,
        os.path.join(OUTPUT_DIR, "mlp_first_layer_weights.png")
    )

    print_error_summary("MLP", cm)


def analyze_cnn():
    cnn_model_path = find_latest_model("./best_models_cnn")
    print("Loading CNN model from:", cnn_model_path)

    model = nn.models.Model_CNN()
    model.load_model(cnn_model_path)

    test_imgs, test_labs = load_mnist_test_for_cnn()

    logits = predict_in_batches(model, test_imgs, batch_size=128)
    preds = np.argmax(logits, axis=1)

    cm = compute_confusion_matrix(test_labs, preds)

    plot_confusion_matrix(
        cm,
        "CNN Confusion Matrix on MNIST",
        os.path.join(OUTPUT_DIR, "cnn_confusion_matrix.png")
    )

    plot_misclassified_examples(
        test_imgs,
        test_labs,
        preds,
        "CNN",
        os.path.join(OUTPUT_DIR, "cnn_misclassified_examples.png"),
        is_cnn=True
    )

    plot_cnn_conv_kernels(
        model,
        os.path.join(OUTPUT_DIR, "cnn_conv1_kernels.png")
    )

    print_error_summary("CNN", cm)


if __name__ == "__main__":
    analyze_mlp()
    analyze_cnn()

    print("Saved all Direction 5 figures in:", OUTPUT_DIR)