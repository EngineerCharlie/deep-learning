import matplotlib.pyplot as plt
import time
import torch
import torchvision
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

plt.show()


class FashionMNIST(d2l.DataModule):
    """The Fashion-MNIST dataset."""

    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True
        )
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True
        )


data = FashionMNIST(resize=(32, 32))
print(len(data.train), len(data.val))

print(data.train[0][0].shape)


@d2l.add_to_class(FashionMNIST)
def text_labels(self, indices):
    """Return text labels."""
    labels = [
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]
    return [labels[int(i)] for i in indices]


@d2l.add_to_class(FashionMNIST)
def get_dataloader(self, train):
    data = self.train if train else self.val
    return torch.utils.data.DataLoader(
        data, self.batch_size, shuffle=train, num_workers=self.num_workers
    )


X, y = next(iter(data.train_dataloader()))
print(X.shape, X.dtype, y.shape, y.dtype)
tic = time.time()
for X, y in data.train_dataloader():
    continue
print(f"{time.time() - tic:.2f} sec")


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images in a grid format."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    for i, (img, ax) in enumerate(zip(imgs, axes)):
        ax.imshow(img.numpy(), cmap="gray")
        ax.axis("off")
        if titles:
            ax.set_title(titles[i], fontsize=9)

    # Hide any extra axes if there are fewer images than slots in the grid
    for i in range(len(imgs), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


@d2l.add_to_class(FashionMNIST)
def visualize(self, batch, nrows=1, ncols=8, labels=[]):
    X, y = batch
    if not labels:
        labels = self.text_labels(y)
    show_images(X.squeeze(1), nrows, ncols, titles=labels)


batch = next(iter(data.val_dataloader()))
data.visualize(batch)
