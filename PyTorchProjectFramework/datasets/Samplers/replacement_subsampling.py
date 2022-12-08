from torch.utils.data import RandomSampler, DataLoader
# from typing import List
# ref https://github.com/pytorch/opacus/blob/64680d16d95b1ec007f56b252c631ff786a4a9a7/opacus/utils/uniform_sampler.py#L22

#
# class UniformWithReplacementSampler(Sampler[List[int]]):
#     r"""
#     This sampler samples elements according to the Sampled Gaussian Mechanism.
#     """
#
#     def __init__(self, *, num_samples: int, generator=None):
#         r"""
#         Args:
#             num_samples: number of samples to draw.
#             generator: Generator used in sampling.
#         """
#         self.num_samples = num_samples
#         self.generator = generator
#
#         if self.num_samples <= 0:
#             raise ValueError(
#                 "num_samples should be a positive integer "
#                 "value, but got num_samples={}".format(self.num_samples)
#             )
#
#     def __len__(self):
#         return int(1 / self.sample_rate)
#
#     def __iter__(self):
#         self.num_batches = int(1 / self.sample_rate)
#         for _ in range(self.num_batches):
#
#             print("mask:",mask)
#             indices = mask.nonzero(as_tuple=False).reshape(-1).tolist()
#             yield indices
#
#             # num_batches -= 1

def replacement_subsampling_trainloader(dataset,num_samples,batch_size):
    sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
    train_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return train_loader