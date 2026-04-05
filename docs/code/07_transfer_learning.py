import torch
from torch import nn
from torchvision import models


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int = 3) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main() -> None:
    device = get_device()
    model = build_model().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("transfer learning model ready")
    print(f"total_params={total_params}")
    print(f"trainable_params={trainable_params}")


if __name__ == "__main__":
    main()
