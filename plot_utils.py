import torch
from matplotlib import pyplot as plt


def plot_roc_curve(fpr, tpr, auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()


def plot_images(data, img_col_name, n_images_col, n_images_row):

    labels_map = {0: "Down", 1: "Up"}

    figure = plt.figure(figsize=(15, 15))
    cols, rows = n_images_col, n_images_row
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, label = data.loc[sample_idx, img_col_name]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label] + str(data.loc[sample_idx, 'date']))
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
