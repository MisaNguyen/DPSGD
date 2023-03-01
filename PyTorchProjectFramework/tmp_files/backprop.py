# # Backprop on the Seeds Dataset
# from random import seed
# from random import randrange
# from random import random
# from csv import reader
# from math import exp
#
# # Load a CSV file
# def load_csv(filename):
#     dataset = list()
#     with open(filename, 'r') as file:
#         csv_reader = reader(file)
#         for row in csv_reader:
#             if not row:
#                 continue
#             dataset.append(row)
#     return dataset
#
# # Convert string column to float
# def str_column_to_float(dataset, column):
#     for row in dataset:
#         row[column] = float(row[column].strip())
#
# # Convert string column to integer
# def str_column_to_int(dataset, column):
#     class_values = [row[column] for row in dataset]
#     unique = set(class_values)
#     lookup = dict()
#     for i, value in enumerate(unique):
#         lookup[value] = i
#     for row in dataset:
#         row[column] = lookup[row[column]]
#     return lookup
#
# # Find the min and max values for each column
# def dataset_minmax(dataset):
#     minmax = list()
#     stats = [[min(column), max(column)] for column in zip(*dataset)]
#     return stats
#
# # Rescale dataset columns to the range 0-1
# def normalize_dataset(dataset, minmax):
#     for row in dataset:
#         for i in range(len(row)-1):
#             row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
#
# # Split a dataset into k folds
# def cross_validation_split(dataset, n_folds):
#     dataset_split = list()
#     dataset_copy = list(dataset)
#     fold_size = int(len(dataset) / n_folds)
#     for i in range(n_folds):
#         fold = list()
#         while len(fold) < fold_size:
#             index = randrange(len(dataset_copy))
#             fold.append(dataset_copy.pop(index))
#         dataset_split.append(fold)
#     return dataset_split
#
# # Calculate accuracy percentage
# def accuracy_metric(actual, predicted):
#     correct = 0
#     for i in range(len(actual)):
#         if actual[i] == predicted[i]:
#             correct += 1
#     return correct / float(len(actual)) * 100.0
#
# # Evaluate an algorithm using a cross validation split
# def evaluate_algorithm(dataset, algorithm, n_folds, *args):
#     folds = cross_validation_split(dataset, n_folds)
#     scores = list()
#     for fold in folds:
#         train_set = list(folds)
#         train_set.remove(fold)
#         train_set = sum(train_set, [])
#         test_set = list()
#         for row in fold:
#             row_copy = list(row)
#             test_set.append(row_copy)
#             row_copy[-1] = None
#         predicted = algorithm(train_set, test_set, *args)
#         actual = [row[-1] for row in fold]
#         accuracy = accuracy_metric(actual, predicted)
#         scores.append(accuracy)
#     return scores
#
# # Calculate neuron activation for an input
# def activate(weights, inputs):
#     activation = weights[-1]
#     for i in range(len(weights)-1):
#         activation += weights[i] * inputs[i]
#     return activation
#
# # Transfer neuron activation
# def transfer(activation):
#     return 1.0 / (1.0 + exp(-activation))
#
# # Forward propagate input to a network output
# def forward_propagate(network, row):
#     inputs = row
#     for layer in network:
#         new_inputs = []
#         for neuron in layer:
#             activation = activate(neuron['weights'], inputs)
#             neuron['output'] = transfer(activation)
#             new_inputs.append(neuron['output'])
#         inputs = new_inputs
#     return inputs
#
# # Calculate the derivative of an neuron output
# def transfer_derivative(output):
#     return output * (1.0 - output)
# # The derivative of relu function is 1 if z > 0, and 0 if z <= 0
# def relu_deriv(output):
#     return 1 * (output > 0)
# # Backpropagate error and store in neurons
# def backward_propagate_error(network, expected):
#     for i in reversed(range(len(network))):
#         layer = network[i]
#         errors = list()
#         if i != len(network)-1:
#             for j in range(len(layer)):
#                 error = 0.0
#                 for neuron in network[i + 1]:
#                     error += (neuron['weights'][j] * neuron['delta'])
#                 errors.append(error)
#         else:
#             for j in range(len(layer)):
#                 neuron = layer[j]
#                 errors.append(neuron['output'] - expected[j])
#         for j in range(len(layer)):
#             neuron = layer[j]
#             neuron['delta'] = errors[j] * relu_deriv(neuron['output'])
#             # neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
#
# # Update network weights with error
# def update_weights(network, row, l_rate):
#     for i in range(len(network)):
#         inputs = row[:-1]
#         if i != 0:
#             inputs = [neuron['output'] for neuron in network[i - 1]]
#         for neuron in network[i]:
#             for j in range(len(inputs)):
#                 neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
#             neuron['weights'][-1] -= l_rate * neuron['delta']
#
# # Train a network for a fixed number of epochs
# def train_network(network, train, l_rate, n_epoch, n_outputs):
#     for epoch in range(n_epoch):
#         for row in train:
#             outputs = forward_propagate(network, row)
#             expected = [0 for i in range(n_outputs)]
#             expected[row[-1]] = 1
#             backward_propagate_error(network, expected)
#             update_weights(network, row, l_rate)
#
# # Initialize a network
# def initialize_network(n_inputs, n_hidden, n_outputs):
#     network = list()
#     hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
#     network.append(hidden_layer)
#     output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
#     network.append(output_layer)
#     return network
#
# # Make a prediction with a network
# def predict(network, row):
#     outputs = forward_propagate(network, row)
#     return outputs.index(max(outputs))
#
# # Backpropagation Algorithm With Stochastic Gradient Descent
# def back_propagation(train, test, l_rate, n_epoch, n_hidden):
#     n_inputs = len(train[0]) - 1
#     n_outputs = len(set([row[-1] for row in train]))
#     network = initialize_network(n_inputs, n_hidden, n_outputs)
#     train_network(network, train, l_rate, n_epoch, n_outputs)
#     predictions = list()
#     for row in test:
#         prediction = predict(network, row)
#         predictions.append(prediction)
#     return(predictions)
#
# # Test Backprop on Seeds dataset
# seed(1)
# # load and prepare data
# filename = 'wheat-seeds.csv'
# dataset = load_csv(filename)
# for i in range(len(dataset[0])-1):
#     str_column_to_float(dataset, i)
# # convert class column to integers
# str_column_to_int(dataset, len(dataset[0])-1)
# # normalize input variables
# minmax = dataset_minmax(dataset)
# normalize_dataset(dataset, minmax)
# # evaluate algorithm
# n_folds = 5
# l_rate = 0.3
# n_epoch = 500
# n_hidden = 5
# scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
# print('Scores: %s' % scores)
# print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))


# -*- coding: utf-8 -*-

import sys
import numpy

numpy.seterr(all='ignore')


'''
activation function
'''
def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))

def dsigmoid(x):
    return x * (1. - x)

def tanh(x):
    return numpy.tanh(x)

def dtanh(x):
    return 1. - x * x

def softmax(x):
    e = numpy.exp(x - numpy.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    else:
        return e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)



'''
Dropout
'''
class Dropout(object):
    def __init__(self, input, label, \
                 n_in, hidden_layer_sizes, n_out, \
                 rng=None, activation=ReLU):

        self.x = input
        self.y = label

        self.hidden_layers = []
        self.n_layers = len(hidden_layer_sizes)

        if rng is None:
            rng = numpy.random.RandomState(1234)

        assert self.n_layers > 0


        # construct multi-layer
        for i in range(self.n_layers):

            # layer_size
            if i == 0:
                input_size = n_in
            else:
                input_size = hidden_layer_sizes[i-1]

            # layer_input
            if i == 0:
                layer_input = self.x

            else:
                layer_input = self.hidden_layers[-1].output()

            # construct hidden_layer
            hidden_layer = HiddenLayer(input=layer_input,
                                       n_in=input_size,
                                       n_out=hidden_layer_sizes[i],
                                       rng=rng,
                                       activation=activation)

            self.hidden_layers.append(hidden_layer)


            # layer for ouput using Logistic Regression (softmax)
            self.log_layer = LogisticRegression(input=self.hidden_layers[-1].output(),
                                                label=self.y,
                                                n_in=hidden_layer_sizes[-1],
                                                n_out=n_out)


    def train(self, epochs=5000, dropout=True, p_dropout=0.5, rng=None):

        for epoch in range(epochs):
            dropout_masks = []  # create     different masks in each training epoch

            # forward hidden_layers
            for i in range(self.n_layers):
                if i == 0:
                    layer_input = self.x

                layer_input = self.hidden_layers[i].forward(input=layer_input)

                if dropout == True:
                    mask = self.hidden_layers[i].dropout(input=layer_input, p=p_dropout, rng=rng)
                    layer_input *= mask

                    dropout_masks.append(mask)


            # forward & backward log_layer
            self.log_layer.train(input=layer_input)


            # backward hidden_layers
            for i in reversed(range(0, self.n_layers)):
                if i == self.n_layers-1:
                    prev_layer = self.log_layer
                else:
                    prev_layer = self.hidden_layers[i+1]

                self.hidden_layers[i].backward(prev_layer=prev_layer)

                if dropout == True:
                    self.hidden_layers[i].d_y *= dropout_masks[i]  # also mask here


    def predict(self, x, dropout=True, p_dropout=0.5):
        layer_input = x

        for i in range(self.n_layers):
            if dropout == True:
                self.hidden_layers[i].W = p_dropout * self.hidden_layers[i].W
                self.hidden_layers[i].b = p_dropout * self.hidden_layers[i].b

            layer_input = self.hidden_layers[i].output(input=layer_input)

        return self.log_layer.predict(layer_input)


'''
Hidden Layer
'''
class HiddenLayer(object):
    def __init__(self, input, n_in, n_out, \
                 W=None, b=None, rng=None, activation=tanh):

        if rng is None:
            rng = numpy.random.RandomState(1234)

        if W is None:
            a = 1. / n_in
            W = numpy.array(rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_in, n_out)))

        if b is None:
            b = numpy.zeros(n_out)  # initialize bias 0

        self.rng = rng
        self.x = input

        self.W = W
        self.b = b

        if activation == tanh:
            self.dactivation = dtanh

        elif activation == sigmoid:
            self.dactivation = dsigmoid

        elif activation == ReLU:
            self.dactivation = dReLU

        else:
            raise ValueError('activation function not supported.')


        self.activation = activation



    def output(self, input=None):
        if input is not None:
            self.x = input

        linear_output = numpy.dot(self.x, self.W) + self.b

        return (linear_output if self.activation is None
                else self.activation(linear_output))


    def sample_h_given_v(self, input=None):
        if input is not None:
            self.x = input

        v_mean = self.output()
        h_sample = self.rng.binomial(size=v_mean.shape,
                                     n=1,
                                     p=v_mean)
        return h_sample


    def forward(self, input=None):
        return self.output(input=input)


    def backward(self, prev_layer, lr=0.1, input=None):
        if input is not None:
            self.x = input

        d_y = self.dactivation(prev_layer.x) * numpy.dot(prev_layer.d_y, prev_layer.W.T)

        self.W += lr * numpy.dot(self.x.T, d_y)
        self.b += lr * numpy.mean(d_y, axis=0)

        self.d_y = d_y


    def dropout(self, input, p, rng=None):
        if rng is None:
            rng = numpy.random.RandomState(123)

        mask = rng.binomial(size=input.shape,
                            n=1,
                            p=1-p)  # p is the prob of dropping

        return mask


'''
Logistic Regression
'''
class LogisticRegression(object):
    def __init__(self, input, label, n_in, n_out):
        self.x = input
        self.y = label
        self.W = numpy.zeros((n_in, n_out))  # initialize W 0
        self.b = numpy.zeros(n_out)          # initialize bias 0


    def train(self, lr=0.1, input=None, L2_reg=0.00):
        if input is not None:
            self.x = input

        p_y_given_x = softmax(numpy.dot(self.x, self.W) + self.b)
        d_y = self.y - p_y_given_x

        self.W += lr * numpy.dot(self.x.T, d_y) - lr * L2_reg * self.W
        self.b += lr * numpy.mean(d_y, axis=0)

        self.d_y = d_y

        # cost = self.negative_log_likelihood()
        # return cost

    def negative_log_likelihood(self):
        sigmoid_activation = softmax(numpy.dot(self.x, self.W) + self.b)

        cross_entropy = - numpy.mean(
            numpy.sum(self.y * numpy.log(sigmoid_activation) +
                      (1 - self.y) * numpy.log(1 - sigmoid_activation),
                      axis=1))

        return cross_entropy


    def predict(self, x):
        return softmax(numpy.dot(x, self.W) + self.b)

    def output(self, x):
        return self.predict(x)


def test_dropout(n_epochs=5000, dropout=True, p_dropout=0.5):

    # XOR
    x = numpy.array([[0,  0],
                     [0,  1],
                     [1,  0],
                     [1,  1]])

    y = numpy.array([[0, 1],
                     [1, 0],
                     [1, 0],
                     [0, 1]])

    rng = numpy.random.RandomState(123)


    # construct Dropout MLP
    classifier = Dropout(input=x, label=y, \
                         n_in=2, hidden_layer_sizes=[10, 10], n_out=2, \
                         rng=rng, activation=ReLU)


    # train
    classifier.train(epochs=n_epochs, dropout=dropout, \
                     p_dropout=p_dropout, rng=rng)


    # test
    print(classifier.predict(x))



if __name__ == "__main__":
    test_dropout()