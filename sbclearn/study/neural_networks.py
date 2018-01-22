'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=ungrouped-imports
from __future__ import division

import math
import random
import matplotlib
from numpy import dot

import matplotlib.pyplot as plt


def step_function(x):
    '''step_function.'''
    return 1 if x >= 0 else 0


def perceptron_output(weights, bias, x):
    '''returns 1 if the perceptron 'fires', 0 if not'''
    return step_function(dot(weights, x) + bias)


def sigmoid(t):
    '''sigmoid.'''
    return 1 / (1 + math.exp(-t))


def neuron_output(weights, inputs):
    '''neuron_output.'''
    return sigmoid(dot(weights, inputs))


def feed_forward(neural_network, input_vector):
    '''takes in a neural network (represented as a list of lists of lists of
    weights) and returns the output from forward-propagating the input'''

    outputs = []

    for layer in neural_network:

        input_with_bias = input_vector + [1]             # add a bias input
        output = [neuron_output(neuron, input_with_bias)  # compute the output
                  for neuron in layer]                   # for this layer
        outputs.append(output)                           # and remember it

        # the input to the next layer is the output of this one
        input_vector = output

    return outputs


def backpropagate(network, input_vector, target):
    '''backpropogate.'''
    outputs = feed_forward(network, input_vector)

    layer = len(network) - 1

    # the output * (1 - output) is from the derivative of sigmoid
    deltas = [output * (1 - output) * (output - target[i])
              for i, output in enumerate(outputs[layer])]

    # adjust weights for output layer (network[-1])
    for i, neuron in enumerate(network[-1]):
        for j, output in enumerate(outputs[layer - 1] + [1]):
            neuron[j] -= deltas[i] * output

    layer -= 1

    # back-propagate errors to hidden layer
    deltas = [output * (1 - output) * dot(deltas, [n[i] for n in network[-1]])
              for i, output in enumerate(outputs[layer])]

    # adjust weights for hidden layer (network[0])
    for i, neuron in enumerate(network[0]):
        for j, inpt in enumerate(input_vector + [1]):
            neuron[j] -= deltas[i] * inpt


def patch(x, y, hatch, color):
    '''return a matplotlib 'patch' object with the specified
    location, crosshatch pattern, and color'''
    return matplotlib.patches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                        hatch=hatch, fill=False, color=color)


def show_weights(network, neuron_idx):
    '''show_wrights.'''
    weights = network[0][neuron_idx]
    abs_weights = map(abs, weights)

    grid = [abs_weights[row:(row + 5)]  # turn the weights into a 5x5 grid
            for row in range(0, 25, 5)]  # [weights[0:5], ..., weights[20:25]]

    ax = plt.gca()  # to use hatching, we'll need the axis

    ax.imshow(grid,  # here same as plt.imshow
              interpolation='none')  # plot blocks as blocks

    # cross-hatch the negative weights
    for i in range(5):  # row
        for j in range(5):  # column
            if weights[5 * i + j] < 0:  # row i, column j = weights[5*i + j]
                # add black and white hatches, so visible whether dark or light
                ax.add_patch(patch(j, i, '/', 'white'))
                ax.add_patch(patch(j, i, '\\', 'black'))
    plt.show()


def make_digit(raw_digit):
    '''make_digit.'''
    return [1 if c == '1' else 0
            for row in raw_digit.split('\n')
            for c in row.strip()]


def predict(network, inpt):
    '''predict.'''
    return feed_forward(network, inpt)[-1]


def main():
    '''main method.'''
    raw_digits = [
        '''11111
             1...1
             1...1
             1...1
             11111''',

        '''..1..
             ..1..
             ..1..
             ..1..
             ..1..''',

        '''11111
             ....1
             11111
             1....
             11111''',

        '''11111
             ....1
             11111
             ....1
             11111''',

        '''1...1
             1...1
             11111
             ....1
             ....1''',

        '''11111
             1....
             11111
             ....1
             11111''',

        '''11111
             1....
             11111
             1...1
             11111''',

        '''11111
             ....1
             ....1
             ....1
             ....1''',

        '''11111
             1...1
             11111
             1...1
             11111''',

        '''11111
             1...1
             11111
             ....1
             11111''']

    inputs = map(make_digit, raw_digits)

    targets = [[1 if i == j else 0 for i in range(10)]
               for j in range(10)]

    num_hidden = 5

    # the network starts out with random weights
    network = [[[random.random() for __ in range(len(inputs[0]) + 1)]
                for _ in range(num_hidden)],
               [[random.random() for __ in range(num_hidden + 1)]
                for _ in range(len(targets[0]))]]

    for _ in range(10000):
        for inpt, target in zip(inputs, targets):
            backpropagate(network, inpt, target)

    for i, inpt in enumerate(inputs):
        outputs = predict(network, inpt)
        print i, [round(p, 2) for p in outputs]

    print
    print [round(x, 2) for x in predict(network,
                                        [0, 1, 1, 1, 0,  # .@@@.
                                         0, 0, 0, 1, 1,  # ...@@
                                         0, 0, 1, 1, 0,  # ..@@.
                                         0, 0, 0, 1, 1,  # ...@@
                                         0, 1, 1, 1, 0]  # .@@@.
                                        )]
    print
    print [round(x, 2) for x in predict(network,
                                        [0, 1, 1, 1, 0,  # .@@@.
                                         1, 0, 0, 1, 1,  # @..@@
                                         0, 1, 1, 1, 0,  # .@@@.
                                         1, 0, 0, 1, 1,  # @..@@
                                         0, 1, 1, 1, 0]  # .@@@.
                                        )]


if __name__ == '__main__':
    main()
