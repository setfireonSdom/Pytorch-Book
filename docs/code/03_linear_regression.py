import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_dataset(num_samples: int = 256) -> TensorDataset:
    torch.manual_seed(42)
    features = torch.randn(num_samples, 1)
    noise = 0.2 * torch.randn(num_samples, 1)
    targets = 3.5 * features + 1.2 + noise
    return TensorDataset(features, targets)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0

    for inputs, targets in loader:
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(loader.dataset)


def main() -> None:
    dataset = make_dataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(1, 21):
        avg_loss = train_one_epoch(model, loader, criterion, optimizer)
        print(f"epoch={epoch:02d} loss={avg_loss:.4f}")

    print(f"weight={model.weight.item():.3f}")
    print(f"bias={model.bias.item():.3f}")


if __name__ == "__main__":
    main()
