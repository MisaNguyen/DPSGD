from torch.utils.data import Dataset
import torch
class NumbersDataset(Dataset):
    def __init__(self):
        self.samples = list(range(1, 1001))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == '__main__':
    dataset = NumbersDataset()
    print(len(dataset))
    print(dataset[100])
    print(dataset[122:361])
    train_kwargs = {'batch_size': 2,  'shuffle': True}
    train_loader = torch.utils.data.DataLoader(dataset[:5], **train_kwargs
                                               )
    epochs = 10
    for epoch in range(epochs):
        print(epoch)
        for data in train_loader:
            print(data)
