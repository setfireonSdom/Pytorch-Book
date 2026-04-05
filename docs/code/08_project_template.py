import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class YourDataset(Dataset):
    def __init__(self) -> None:
        self.features = torch.randn(64, 20)
        self.labels = torch.randint(0, 3, (64,))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class YourModel(nn.Module):
    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, num_classes: int = 3) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(loader.dataset)


def main() -> None:
    device = get_device()
    dataset = YourDataset()
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = YourModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 6):
        loss = train_one_epoch(model, loader, criterion, optimizer, device)
        print(f"epoch={epoch} loss={loss:.4f}")


if __name__ == "__main__":
    main()
