import torch


def inspect_tensor(name: str, value: torch.Tensor) -> None:
    print(f"{name}:")
    print(value)
    print(f"shape={tuple(value.shape)}, dtype={value.dtype}")
    print("-" * 40)


def main() -> None:
    x = torch.arange(12, dtype=torch.float32)
    inspect_tensor("x", x)

    matrix = x.reshape(3, 4)
    inspect_tensor("matrix", matrix)

    batch = matrix.unsqueeze(0)
    inspect_tensor("batch", batch)

    flattened = batch.flatten(start_dim=1)
    inspect_tensor("flattened", flattened)


if __name__ == "__main__":
    main()
