#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 18:09:00 2017

@author: kasperipalkama
"""
from random import random, seed
from math import exp

def activation(x):
    '''logistic function as activation function
    '''
    return 1.0 / (1.0 + exp(-x))

def sigma(neuron,inputs,bias):
    return sum([inputs[i]*neuron['weights'][i] \
                      for i in range(len(inputs))]) + bias

def neuronOutput(layer,inputs):
    '''computes the output from each neuron 
    
    -Parameters:
        layer: layer of the network as list.
        inputs: list of inputs for the layer.
    
    -Returns:
        output of the layer as list.
    '''
    activated_outputs = list()
    for neuron in layer:
        bias = neuron['weights'][-1]
        output = sigma(neuron, inputs, bias)
        activated_output = activation(output)
        neuron['output'] = activated_output
        activated_outputs.append(activated_output)
    return activated_outputs

def feedforward(network,inputs):
    '''feedforward for network
    
    -Parameters:
        layer: network weights as list of dictionaries.
        inputs: list of inputs for the layer.
    
    -Returns:
        output of the whole network.
    '''
    new_inputs = inputs
    for layer in network:
        new_inputs = neuronOutput(layer,new_inputs)
    return new_inputs

def derivativeActivation(x):
    '''derivative of logistic function
    '''
    return x * (1.0 - x)
    
def backpropagate(network, desired_outputs):
    '''Backpropagates through network and stores backpropagation error for
       each neuron.
       
    -Parameters:
        network: forward propagated network.
        desired _outputs: learning target as list.
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
            neuron['error'] = costs[j]*derivativeActivation(neuron['output'])

    
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


class FF_ANN_backpropagation:
    
    
    def __init__(self,n_input, n_hidden_neuron, n_hidden_layer, n_output=2):
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
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden_neuron = n_hidden_neuron
        self.n_hidden_layer = n_hidden_layer
        
        network = list()
        for i in range(self.n_hidden_layer):
            if i == 0:
                hiddenLayer = [{'weights': [random() for j in range(self.n_input+1)]} \
                               for j in range(self.n_hidden_neuron)]
            else:
                hiddenLayer = [{'weights': [random() for j in range(self.n_hidden_neuron+1)]} \
                               for j in range(self.n_hidden_neuron)]
            network.append(hiddenLayer)
        outputLyer = [{'weights': [random() for j in range(self.n_hidden_neuron+1)]}\
                      for j in range(self.n_output)]
        network.append(outputLyer)
        self.network = network
        
    
    def train(self, data, rate, n_epochs, print_learning = False):
        '''training of the network
        
        -Parameters:
            network: initialized network.
            date: learning data where targets are in the last column.
            rate: learning rate.
            n_epochs: the number of epochs.
            n_outputs: the number of outputs.
    
        '''
        for epoch in range(n_epochs):
            error_sum = 0
            for row in data:
                desired_output = [0 for i in range(self.n_output)]
                desired_output[int(row[-1])] = 1
                outputs = feedforward(self.network,inputs=row)
                error_sum += errorCalc(desired_output,outputs)
                backpropagate(self.network, desired_output)
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
            for row in data:
                output = feedforward(self.network,row)
                if pred_prob:
                    predictions.append([output[i]/sum(output) for i in range(self.n_output)])
                else:
                    predictions.append(output.index(max(output)))
            self.predicted = predictions
            return predictions
        else:
            raise Exception('Network is not trained!')
            
    def score(self,trueClasses):
        n_equal = 0
        for true, pred in zip(trueClasses, self.predicted):
            if true == pred:
                n_equal += 1
        return n_equal / len(trueClasses)
    
        