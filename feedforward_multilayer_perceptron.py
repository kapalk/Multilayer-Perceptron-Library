#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 18:16:14 2017

@author: kasperipalkama
"""
from random import uniform, seed
from math import exp

def activation(x,function):
    '''returns activation function according to user input
    '''
    logistic = 1.0 / (1.0 + exp(-x))
    tanh = (exp(2*x)-1) / (exp(2*x)+1)
    rluf = max(0,x)
    functions = {'logistic': logistic , 'tanh': tanh, 'rluf': rluf }
    try:
        return functions[function]
    except KeyError:
        print('Current activation function is not supported')


def summing_junction(neuron,inputs,bias):
    return sum([inputs[i]*neuron['weights'][i] \
                      for i in range(len(inputs))]) + bias

def neuronOutput(layer,inputs,activation_func):
    '''computes the output from each neuron

    -Parameters:
        layer: layer of the network as list.
        inputs: list of inputs for the layer.
        activation_func: activation function as string
    -Returns:
        output of the layer as list.
    '''
    outputs = list()
    for neuron in layer:
        bias = neuron['weights'][-1]
        lin_response = summing_junction(neuron, inputs, bias)
        output = activation(lin_response,activation_func)
        neuron['output'] = output
        outputs.append(output)
    return outputs

def feedforward(network,inputs, activation_func):
    '''feedforward for network

    -Parameters:
        layer: network weights as list of dictionaries.
        inputs: list of inputs for the layer.
        activation_func: activation function as string

    -Returns:
        output of the whole network.
    '''
    new_inputs = inputs
    for layer in network:
        new_inputs = neuronOutput(layer,new_inputs,activation_func)
    return new_inputs

def derivativeActivation(x, function):
    '''derivative of logistic function
    '''
    logistic_derivative = x * (1.0 - x)
    tanh_derivative = 4 /((exp(-x) + exp(x))**2)
    functions = {'logistic': logistic_derivative, 'tanh': tanh_derivative}
    return functions[function]

def backpropagate(network, desired_outputs, activation_func):
    '''Backpropagates through network and stores backpropagation error for
       each neuron.

    -Parameters:
        network: forward propagated network.
        desired _outputs: learning target as list.
        activation_func: activation function as string
    '''

    for i in reversed(range(len(network))):
        layer = network[i]
        costs = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                cost = 0.0
                inputLayer = network[i+1]
                for neuron in inputLayer:
                    cost += neuron['weights'][j] * neuron['error']
                costs.append(cost)

        else:
            for j, neuron in enumerate(layer):
                cost = desired_outputs[j] - neuron['output']
                costs.append(cost)
        for j, neuron in enumerate(layer):
            neuron['error'] = costs[j]*derivativeActivation(neuron['output'],
                                                          activation_func)


def updateWeights(network, rate, input_list):
    '''Updates weights according in following:
        weight = weight + learning_rate * error * input

    -Parameters:
        network: backpropagated network.
        rate: learning rate.
        input_list: training data instance as list

    '''
    for ix,layer in enumerate(network):
        if ix == 0:
            inputs = input_list[:-1]
        else:
            inputs = [neuron['output'] for neuron in network[ix-1]]
        for neuron in layer:
            for i in range(len(inputs)):
                neuron['weights'][i] += rate * neuron['error']*inputs[i]
            neuron['weights'][-1] +=  rate * neuron['error']

def errorCalc(desired_outputs, outputs):
    '''computes training error for single training sample
    '''
    return sum([(desired_outputs[i]-outputs[i])**2 \
                 for i in range(len(desired_outputs))])


class MLP_classifier:


    def __init__(self,n_input, n_hidden_neuron, n_hidden_layer = 2, n_output=2,
                 activation = 'logistic'):
        '''initializes the network

        -Parameters:
            n_input: the number of inputs.
            n_hidden: the number of neurons in hidden layers.
            n_hidden_layer: the number of hidden layers.
            n_outputs: the number of outputs.

        '''
        seed(1)
        self.trained = False
        self.predicted = None
        self.predicted_prob = None
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden_neuron = n_hidden_neuron
        self.n_hidden_layer = n_hidden_layer
        self.activation = activation
        network = list()
        for i in range(self.n_hidden_layer):
            if i == 0:
                left_bound = -0.5 / n_input # common way to initilize weights
                rigth_bound = 0.5 / n_input
                hiddenLayer = [{'weights': [uniform(left_bound,rigth_bound) \
                                for j in range(self.n_input+1)]} \
                               for j in range(self.n_hidden_neuron)]
            else:
                left_bound = -0.5 / n_hidden_neuron
                rigth_bound = 0.5 / n_hidden_neuron
                hiddenLayer = [{'weights': [uniform(left_bound,rigth_bound) \
                                for j in range(self.n_hidden_neuron+1)]} \
                               for j in range(self.n_hidden_neuron)]
            network.append(hiddenLayer)
        outputLyer = [{'weights': [uniform(left_bound,rigth_bound) \
                                   for j in range(self.n_hidden_neuron+1)]}\
                      for j in range(self.n_output)]
        network.append(outputLyer)
        self.network = network


    def train(self, data, learning_rate_init, n_epochs, learning_rate = 'constant',
     print_learning = False):
        '''training of the network

        -Parameters:
            network: initialized network.
            date: learning data where targets are in the last column.
            rate: learning rate.
            n_epochs: the number of epochs.
            n_outputs: the number of outputs.

        '''
        iter = 1
        for epoch in range(n_epochs):
            error_sum = 0
            for row in data:
                iter += 1
                desired_response = [0 for i in range(self.n_output)]
                desired_response[int(row[-1])] = 1
                outputs = feedforward(self.network,row,self.activation)
                error_sum += errorCalc(desired_response,outputs)
                backpropagate(self.network, desired_response, self.activation)
                if learning_rate == 'decreasive':
                    rate = learning_rate_init / (1 + iter / 100)
                else:
                    rate = learning_rate_init
                updateWeights(self.network, rate, row)
            if print_learning:
                print('epoch=%d, error=%.3f' % (epoch, error_sum))
        self.trained = True


    def predict(self, data, pred_prob = False):
        '''Predicts targets

        -Parameters:
            network: trained network.
            data: features.
            n_output: the number of classes:
            pred_prob: whether to predict crips classes or probabilites

        -Returns:
            list of predictions.
        '''
        if self.trained:
            predictions = list()
            predictions_prob = list()
            for row in data:
                output = feedforward(self.network,row, self.activation)
                print(output)
                if self.activation == 'logistic':
                    predictions_prob.append([output[i] / sum(output)
                    for i in range(self.n_output)])
                if self.activation == 'tanh':
                    predictions_prob.append([abs(output[i])/sum(map(abs,output))
                    for i in range(self.n_output)])
                predictions.append(output.index(max(map(abs,output))))
            self.predicted = predictions
            self.predicted_prob = predictions_prob
            if pred_prob == False:
                return predictions
            else:
                return predictions_prob
        else:
            raise Exception('Network is not trained!')

    def score(self,trueClasses):
        n_equal = 0
        for true, pred in zip(trueClasses, self.predicted):
            if true == pred:
                n_equal += 1
        return n_equal / len(trueClasses)
