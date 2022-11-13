from torch.utils.data import Dataset
import torch
from numpy.random import choice

from tmp_files.opacus_sampler import DistributedUniformWithReplacementSampler, UniformWithReplacementSampler

class NumbersDataset(Dataset):
    def __init__(self):
        self.samples = list(range(1, 1001))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == '__main__':
    dataset = NumbersDataset()
    # print(len(dataset))
    # print(dataset[100])
    # print(dataset[122:361])
    ### shuflling method
    train_kwargs = {'batch_size': 3,  'shuffle': True} # shuffle whole dataset

    sampling_method = "subsampling"
    sample_rate = 1/2 # = batch_size / len(dataset)
    epochs = 10

    number_of_batches = len(dataset[:10]) // train_kwargs['batch_size']
    batches_length = [train_kwargs['batch_size']] * number_of_batches
    """ No drop last option"""
    if len(dataset[:10]) % train_kwargs['batch_size'] != 0:
        batches_length.append(len(dataset[:10]) % train_kwargs['batch_size'])
    """ Split train dataset into batches"""
    batches = torch.utils.data.random_split(dataset[:10], batches_length)
    for batch_idx , batch in enumerate(batches):
        print("id:",batch_idx)
        for data in batch:
            print(data)
    # generator = torch.Generator()
    # batch_sampler = DistributedUniformWithReplacementSampler(
    #     total_size=len(dataset),  # type: ignore[assignment, arg-type]
    #     sample_rate=sample_rate,
    #     generator=None,
    # )
    # batch_sampler = UniformWithReplacementSampler(
    #     num_samples=10,  # type: ignore[assignment, arg-type]
    #     sample_rate=sample_rate,
    #     generator=None,
    # )
    # if(sampling_method == "subsampling"):
    #     train_loader = torch.utils.data.DataLoader(dataset[:10], batch_sampler=batch_sampler,
    #                                                )
    # else:
    #     train_loader = torch.utils.data.DataLoader(dataset[:10],
    #                                                **train_kwargs
    #                                                )
    # for epoch in range(epochs):
    #     print("E:",epoch)
    #     # print("loader:", train_loader)
    #     # print(len(train_loader))
    #     for batch_idx, data in enumerate(train_loader): # Batch loop
    #
    #             # Uniformly pick indexes, ref: https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
    #             # Replace = true => index can appear multiple time,
    #             # p = None => Uniform distribution
    #             # Size = None => Output single value
    #             # index= choice(len(train_loader), size=None, replace=True, p=None)
    #         # data = train_loader
    #         print(batch_idx)
    #         # print(data_x)
    #         print(data)
