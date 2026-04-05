import torch
from torch.utils.data import DataLoader, Dataset


class StudentScoreDataset(Dataset):
    def __init__(self) -> None:
        self.features = torch.tensor(
            [
                [4.0, 1.0, 0.0],
                [6.0, 0.0, 1.0],
                [8.0, 1.0, 1.0],
                [3.0, 0.0, 0.0],
                [9.0, 1.0, 1.0],
                [5.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self.labels = torch.tensor([0, 1, 1, 0, 1, 0], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def main() -> None:
    dataset = StudentScoreDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch_idx, (inputs, labels) in enumerate(loader, start=1):
        print(f"batch {batch_idx}")
        print("inputs shape:", tuple(inputs.shape))
        print("labels shape:", tuple(labels.shape))
        print(inputs)
        print(labels)
        print("-" * 40)


if __name__ == "__main__":
    main()
