import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import datasets, transforms
class NumbersDataset(Dataset):
    def __init__(self):
        self.samples = list(range(1, 1001))
        # self.labels =
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear1 = torch.nn.Linear(inputSize, 1)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(1, 1)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(1, outputSize)
        self.relu3 = torch.nn.ReLU()
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        return x

if __name__ == '__main__':
    # toTensor = [
    #     transforms.ToTensor(),
    # ]
    # augmentations = [
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    # ]
    #
    # train_normalize = [
    #     transforms.Normalize((0.1307,), (0.3081,)),
    #     #transforms.Normalize(test_mean, test_std),
    # ]
    # test_normalize = [
    #     transforms.Normalize((0.1307,), (0.3081,)),
    #     #transforms.Normalize(test_mean, test_std),
    # ]
    # transform_test = transforms.Compose(
    #     augmentations + toTensor
    #     + test_normalize
    # )
    # transform_train = transforms.Compose(
    #     augmentations + toTensor
    #     + train_normalize
    # )
    # testset = datasets.MNIST(
    #     root='../data', train=False, download=True, transform=transform_test)
    # # testset = MyDataset(testset,transform=transform_test)
    #
    # trainset = datasets.MNIST(
    #     root='../data', train=True, download=True, transform=transform_train)
    #
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=1)
    # print("Finished normalizing dataset.")
    # create dummy data for training
    x_values = [i for i in range(11)]
    x_train = np.array(x_values, dtype=np.float32)
    x_train = x_train.reshape(-1, 1)

    y_values = [2*i + 1 for i in x_values]
    for i in range(len(y_values)):
        random = 10* np.random.rand(1)
        y_values[i] = y_values[i] + random
    y_train = np.array(y_values, dtype=np.float32)
    y_train = y_train.reshape(-1, 1)
    # print("x_train = ", x_train)
    # print("y_train = ", y_train)
    # input()
    inputDim = 11   # takes variable 'x'
    outputDim = 10      # takes variable 'y'
    learningRate = 0.01
    epochs = 10

    model = linearRegression(inputDim, outputDim)
    ##### For GPU #######
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    for epoch in range(epochs):
        print("epoch = ", epoch)
        # # Converting inputs and labels to Variable
        # for (inputs, labels) in train_loader:
        #     inputs,labels = inputs.to(device),labels.to(device)
        if torch.cuda.is_available():
            inputs = Variable(torch.from_numpy(x_train).cuda())
            labels = Variable(torch.from_numpy(y_train).cuda())
            # inputs,labels = trainset.cuda()
        else:
            inputs = Variable(torch.from_numpy(x_train))
            labels = Variable(torch.from_numpy(y_train))
            # inputs,labels = trainset

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
            optimizer.zero_grad()

            # get output from the model, given the inputs
            outputs = model(inputs)

            # get loss for the predicted output
            loss = criterion(outputs, labels)
            print(loss)
            # get gradients w.r.t to parameters
            loss.backward()
            for name, param in model.named_parameters():
                print("layer:", name)
                print("grad:", param.grad)
            # update parameters
            optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))