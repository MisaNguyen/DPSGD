from datasets.base_dataset import get_transform
from datasets.base_dataset import BaseDataset
import torch

from skimage.io import imread
from torch.utils import data

class Segmentation2DDataset(BaseDataset):
    """Represents a 2D segmentation dataset.
    
    Input params:
        configuration: Configuration dictionary.
    """
    def __init__(self, configuration):
        # self.inputs = inputs
        # self.targets = targets
        # self.transform = transform
        # self.inputs_dtype = torch.float32
        # self.targets_dtype = torch.long
        print(configuration)
        super().__init__(configuration)


    def __getitem__(self, index):
        # get source image as x
        # get labels as y

        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # Load input and target
        x, y = imread(input_ID), imread(target_ID)

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return (x, y)

    def __len__(self):
        # return the size of the dataset
        return len(self.inputs)
