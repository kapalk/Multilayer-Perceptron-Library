#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 18:16:14 2017

@author: kasperipalkama
"""
from random import random
from math import exp

def initializeNetwork(n_input, n_hidden, n_hidden_layer, n_output):
    '''initializes random weights for each neuron
    
    -Parameters: 
        n_input: the number of inputs.
        n_hidden: the number of neurons in hidden layers.
        n_hidden_layer: the number of hidden layers.
        n_outputs: the number of outputs.
        
    -Returns:
        list of dictionaries in which each dict contains 
        weights of the layer.

    '''
    network = list()
    for i in range(n_hidden_layer):
        if i == 0:
            hiddenLayer = [{'weights': [random() for j in range(n_input+1)]} \
                           for j in range(n_hidden)]
        else:
            hiddenLayer = [{'weights': [random() for j in range(n_hidden+1)]} \
                           for j in range(n_hidden)]
        network.append(hiddenLayer)
    outputLyer = [{'weights': [random() for j in range(n_hidden+1)]}\
                  for j in range(n_output)]
    network.append(outputLyer)
    return network

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
    for i in reversed(range(len(network))):
        layer = network[i]
        costs = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                cost = 0.0
                inputLayer = network[i+1]
                for neuron in inputLayer:
                    cost += neuron['weights'][j] * neuron['delta']
                costs.append(cost)
            
        else:
            for j, neuron in enumerate(layer):
                cost = desired_outputs[j] - neuron['output']
                costs.append(cost)
        for j, neuron in enumerate(layer):
            neuron['delta'] = costs[j]*derivativeActivation(neuron['output'])
    
    
                
if __name__ == '__main__':
    network = initializeNetwork(n_input=3,n_hidden=2,
                                 n_hidden_layer=5,n_output=2)
    finaloutputfeedforward = feedforward(network,[1,1,1])    
    expected = [0,1]
    backpropagate(network,expected)
