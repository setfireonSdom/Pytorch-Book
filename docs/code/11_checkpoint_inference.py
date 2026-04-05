from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    checkpoint_path = Path("artifacts/fashion_mnist_cnn.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint not found. Run code/10_image_classification_project.py first."
        )

    device = get_device()
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    test_dataset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )
    loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    images, labels = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        logits = model(images)
        predictions = logits.argmax(dim=1).cpu()

    for idx, (pred, label) in enumerate(zip(predictions, labels), start=1):
        pred_name = CLASS_NAMES[pred.item()]
        label_name = CLASS_NAMES[label.item()]
        print(f"sample={idx} pred={pred_name} target={label_name}")


if __name__ == "__main__":
    main()
