import torch


def main() -> None:
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print("x:")
    print(x)
    print("shape:", x.shape)

    x_batch = x.unsqueeze(0)
    print("after unsqueeze:", x_batch.shape)

    w = torch.tensor(2.0, requires_grad=True)
    y = (w * x.mean() - 3) ** 2
    y.backward()

    print("loss:", y.item())
    print("gradient of w:", w.grad.item())


if __name__ == "__main__":
    main()
