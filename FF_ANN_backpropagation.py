#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 18:16:14 2017

@author: kasperipalkama
"""
from random import random
from math import exp

def initialize_network(n_input, n_hidden, n_hidden_layer, n_output):
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

def neuronOutput(layer,inputs):
    '''computes the output from each neuron 
    
    -Parameters:
        layer: layer of the network as list.
        inputs: list of inputs for the layer.
    
    -Returns:
        output of the neuron as list.
    '''
    outputs = list()
    for neuron in layer:
        outputs.append(sum([inputs[i]*neuron['weights'][i] \
                      for i in range(len(inputs))]) + neuron['weights'][-1])
    return outputs
        
def activation(x):
    '''logistic function as activation function
    '''
    return 1 / (1 + exp(x))
    
def feedforward(network,inputs):
    '''feedforward for network
    
    -Parameters:
        layer: network weights as list of dictionaries.
        inputs: list of inputs for the layer.
    
    -Returns:
        output of the whole network.
    '''
    for layer in network:
        inputs = neuronOutput(layer,inputs)
        inputs = [activation(neuronInput) for neuronInput in inputs]
    return inputs
        
        
    
if __name__ == '__main__':
    network = initialize_network(n_input=3,n_hidden=2,
                                 n_hidden_layer=2,n_output=2)
    finaloutputfeedforward(network,[1,1,1]))
    backprogate()
    
    