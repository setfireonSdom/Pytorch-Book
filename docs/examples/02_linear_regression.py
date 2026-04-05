import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def build_dataset(num_samples: int = 256) -> TensorDataset:
    torch.manual_seed(42)
    x = torch.randn(num_samples, 1)
    noise = 0.2 * torch.randn(num_samples, 1)
    y = 3.5 * x + 1.2 + noise
    return TensorDataset(x, y)


def main() -> None:
    dataset = build_dataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(1, 21):
        epoch_loss = 0.0
        for inputs, targets in loader:
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        avg_loss = epoch_loss / len(dataset)
        print(f"epoch={epoch:02d} loss={avg_loss:.4f}")

    weight = model.weight.item()
    bias = model.bias.item()
    print(f"learned weight={weight:.3f}, bias={bias:.3f}")


if __name__ == "__main__":
    main()
