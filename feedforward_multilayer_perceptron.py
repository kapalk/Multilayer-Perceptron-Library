#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 18:16:14 2017

@author: kasperipalkama
"""
from random import uniform, seed
from math import exp

def initialize(n_input, n_hidden_neuron, n_hidden_layer, n_output):
    seed(1)
    network = list()
    for i in range(n_hidden_layer):
        if i == 0:
            left_bound = -0.5 / n_input # common way to initilize weights
            rigth_bound = 0.5 / n_input
            hiddenLayer = [{'weights': [uniform(left_bound,rigth_bound) \
                            for j in range(n_input+1)]} \
                           for j in range(n_hidden_neuron)]
        else:
            left_bound = -0.5 / n_hidden_neuron
            rigth_bound = 0.5 / n_hidden_neuron
            hiddenLayer = [{'weights': [uniform(left_bound,rigth_bound) \
                            for j in range(n_hidden_neuron+1)]} \
                           for j in range(n_hidden_neuron)]
        network.append(hiddenLayer)
    outputLayer = [{'weights': [uniform(left_bound,rigth_bound) \
                               for j in range(n_hidden_neuron+1)]}\
                  for j in range(n_output)]
    network.append(outputLayer)
    return network

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

def neuronOutput(layer,inputs,activation_func, regression, output_layer):
    '''computes the output from each neuron. If regression, output layer is linear

    -Parameters:
        layer: layer of the network as list.
        inputs: list of inputs for the layer.
        activation_func: activation function as string
    -Returns:
        output of the layer as list.
    '''
    outputs = list()
    for i,neuron in enumerate(layer):
        bias = neuron['weights'][-1]
        lin_response = summing_junction(neuron, inputs, bias)
        if regression and output_layer:
            output = lin_response
        else:
            output = activation(lin_response,activation_func)
        neuron['output'] = output
        outputs.append(output)
    return outputs

def feedforward(network,inputs, activation_func, regression = False):
    '''feedforward for network

    -Parameters:
        layer: network weights as list of dictionaries.
        inputs: list of inputs for the layer.
        activation_func: activation function as string

    -Returns:
        output of the whole network.
    '''
    new_inputs = inputs
    output_layer = False
    for i, layer in enumerate(network):
        if i+1 == len(network):
            output_layer = True
        new_inputs = neuronOutput(layer,new_inputs,activation_func, 
                                  regression, output_layer)
    if regression:
        return new_inputs[0]
    else:
        return new_inputs

def derivativeActivation(x, function):
    '''derivative of logistic function
    '''
    logistic_derivative = x * (1.0 - x)
    tanh_derivative = 4 /((exp(-x) + exp(x))**2)
    functions = {'logistic': logistic_derivative, 'tanh': tanh_derivative}
    return functions[function]

def backpropagate(network, desired_outputs, activation_func, regression):
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
                outerLayer = network[i+1]
                for neuron in outerLayer:
                    cost += neuron['weights'][j] * neuron['delta']
                costs.append(cost)

        else:
            for desired_output, neuron in zip(desired_outputs, layer):
                cost = desired_output - neuron['output']
                costs.append(cost)
        for neuron, cost in zip(layer, costs):
            if regression and i==len(network)-1:
                neuron['delta'] = cost
            else:
                neuron['delta'] = cost*derivativeActivation(neuron['output'],
                                                          activation_func)


def updateWeights(network, rate, input_list):
    '''Updates weights according in following:
        weight = weight + learning_rate * delta * input

    -Parameters:
        network: backpropagated network.
        rate: learning rate.
        input_list: training data instance as list

    '''
    for i,layer in enumerate(network):
        if i == 0:
            inputs = input_list[:-1]
        else:
            inputs = [neuron['output'] for neuron in network[i-1]]
        for neuron in layer:
#            inputs = [inputs[0] for i in range(len(neuron['weights'])-1)]
            for j in range(len(inputs)):
                neuron['weights'][j] += rate * neuron['delta']*inputs[j]
            neuron['weights'][-1] +=  rate * neuron['delta']

def errorCalc(desired_outputs, outputs, regression = False):
    '''computes training error for single training sample
    '''
    if regression:
        outputs = [outputs]
    return sum([(desired_outputs[i]-outputs[i])**2 \
                 for i in range(len(desired_outputs))])


def trainPredictor(network, data, learning_rate_init, 
                   n_epochs, n_output, activation = 'logistic', 
                   learning_rate = 'constant', print_learning = False):
    '''training of the network

        -Parameters:
            network: initialized network.
            data: learning data where targets are in the last column.
            learning_rate_init: initial learning rate.
            n_epochs: the number of epochs.
            n_outputs: the number of outputs
            activation: chosen activation function
            learning_rate: learning rate adjusting method
            print_learning: is learning printed 
    '''
    regression = False
    if n_output == 1:
        regression = True
    iter_count = 1
    for epoch in range(n_epochs):
        error_sum = 0
        for row in data:
            iter_count += 1
            if regression:
                desired_response = [row[-1]]
            else:
                desired_response = [0 for i in range(n_output)]
                desired_response[int(row[-1])] = 1
            outputs = feedforward(network,row,activation,
                                  regression=regression)
            error_sum += errorCalc(desired_response, outputs, regression)
            backpropagate(network, desired_response, activation, regression)
            if learning_rate == 'decreasive':
                rate = learning_rate_init / (1 + iter_count / 100)
            else:
                rate = learning_rate_init
            updateWeights(network, rate, row)
        if print_learning:
            print('epoch=%d, error=%.3f' % (epoch, error_sum))
    return network, True

def predictValues(trained, network, data, n_output, activation, 
                  pred_prob = False):
    '''Predicts targets

        -Parameters:
            trained: boolean value
            network: trained network.
            data: features.
            n_output: the number of outputs:
            activation: activation function used
            pred_prob: whether to predict crips classes or probabilites
        -Returns:
            list of predictions.
    '''
    if trained:
        predictions = list()
        predictions_prob = list()
        for row in data:
            output = feedforward(network, row, activation)
            if n_output > 1:
                if activation == 'logistic':
                    predictions_prob.append([output[i] / sum(output)
                    for i in range(n_output)])
                if activation == 'tanh':
                    predictions_prob.append([abs(output[i])/sum(map(abs,output))
                    for i in range(n_output)])
            predictions.append(output.index(max(map(abs,output))))
        if not pred_prob:
            return predictions
        else:
            return predictions_prob
    else:
        raise Exception('Network is not trained!')

class MLP_classifier:


    def __init__(self,n_input, n_hidden_neuron, n_hidden_layer = 2, n_output=2,
                 activation = 'logistic'):
        '''initializes the network

        -Parameters:
            n_input: the number of inputs.
            n_hidden: the number of neurons in hidden layers.
            n_hidden_layer: the number of hidden layers.
            n_outputs: the number of outputs.
            activation: activation function on each neuron.

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
        self.network = initialize(n_input, n_hidden_layer, n_hidden_neuron,
                                  n_output)    

    def train(self, data, learning_rate_init, n_epochs, learning_rate = 'constant',
     print_learning = False):
        '''training of the network

        -Parameters:
            data: learning data where targets are in the last column.
            learning_rate_init: initial learning rate.
            n_epochs: the number of epochs.
            learning_rate: learning rate adjusting method
            print_learning: is learning printed 
        '''
        self.network, self.trained = trainPredictor(self.network, data, 
                                                    learning_rate_init, 
                                                    n_epochs, self.n_output, 
                                                    self.activation, 
                                                    learning_rate, 
                                                    print_learning)


    def predict(self, data, pred_prob = False):
        '''Predicts targets

        -Parameters:
            network: trained network.
            data: features.
            pred_prob: whether to predict crips classes or probabilites
        -Returns:
            list of predictions.
        '''
        if pred_prob:
            self.predictions = predictValues(self.trained, self.network, 
                                             data, self.n_output, 
                                             self.activation, pred_prob)
            return self.predictions
        else:
            self.predicted_prob = predictValues(self.trained, self.network, 
                                                data, self.n_output, 
                                                self.activation, pred_prob)
            return self.predicted_prob
    
       

    def score(self,trueClasses):
        n_equal = 0
        for true, pred in zip(trueClasses, self.predicted):
            if true == pred:
                n_equal += 1
        return n_equal / len(trueClasses)

class MLP_regressor:
    
    def __init__(self,n_input, n_hidden_neuron, n_hidden_layer, activation):
        '''initializes the network

        -Parameters:
            n_input: the number of inputs.
            n_hidden: the number of neurons in hidden layers.
            n_hidden_layer: the number of hidden layers.
            activation: activation function on each neuron.
        '''
        seed(1)
        self.trained = False
        self.predictions = None
        self.n_input = n_input
        self.n_hidden_neuron = n_hidden_neuron
        self.n_hidden_layer = n_hidden_layer
        self.n_output = 1
        self.activation = activation
        self.network = initialize(n_input, n_hidden_layer, n_hidden_neuron, 1)
        
    def train(self, data, learning_rate_init, n_epochs, learning_rate = 'constant',
     print_learning = False):
        '''training of the network

        -Parameters:
            network: initialized network.
            date: learning data where targets are in the last column.
            rate: learning rate.
            n_epochs: the number of epochs.

        '''
        self.network, self.trained = trainPredictor(self.network, data, 
                                                    learning_rate_init, 
                                                    n_epochs, self.n_output, 
                                                    self.activation, 
                                                    learning_rate, 
                                                    print_learning)
    
    def predict(self, data):
        '''Predicts targets

        -Parameters:
            network: trained network.
            data: features.
        -Returns:
            list of predictions.
        '''
        self.predictions = predictValues(self.trained, self.network, 
                                             data, self.n_output, 
                                             self.activation, 
                                             pred_prob=False)
        return self.predictions
    
    def score_mse(self,trueValues):
        '''prediction performance as mean squared error
        '''
        n_samples = len(trueValues)
        total_squared_error = 0
        for true, predicted in zip(trueValues, self.predictions):
            total_squared_error += (true-predicted)**2
        return 1/n_samples * total_squared_error
        
        