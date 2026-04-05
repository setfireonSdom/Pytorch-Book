import torch


def main() -> None:
    weight = torch.tensor(2.0, requires_grad=True)
    bias = torch.tensor(1.0, requires_grad=True)

    x = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([4.0, 7.0, 10.0])

    y_pred = weight * x + bias
    loss = ((y_pred - y_true) ** 2).mean()

    loss.backward()

    print(f"loss={loss.item():.4f}")
    print(f"weight.grad={weight.grad.item():.4f}")
    print(f"bias.grad={bias.grad.item():.4f}")


if __name__ == "__main__":
    main()
