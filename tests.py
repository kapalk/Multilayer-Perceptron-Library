#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:31:18 2017

@author: kasperipalkama
"""

from feedforward_multilayer_perceptron import initialize, feedforward, backpropagate, updateWeights
import unittest


class tests(unittest.TestCase):
    
    
    def test_initialize(self):
        '''tests networks data types and structure
        '''
        n_input = 5
        n_hidden_neuron = 10
        n_hidden_layer = 3
        n_output = 3
        
        net = initialize(n_input, n_hidden_neuron, n_hidden_layer, n_output)
        
        self.assertIsInstance(net,list,msg='network data type error')
        self.assertEqual(len(net),n_hidden_layer+1,msg='layer count error')
        for i in range(len(net)):
            self.assertIsInstance(net[i],list,msg='layer data type error')
            n_neuron = len(net[i])
            if i == len(net)-1:
                self.assertEqual(n_neuron, n_output,
                                 msg='output neuron count error')
            else:
                self.assertEqual(n_neuron, n_hidden_neuron,
                                 msg='input/hidden neuron count error')
            for j in range(len(net[i])):
                self.assertIsInstance(net[i][j],dict, 
                                      msg='neuron data type error')
                self.assertIsInstance(net[i][j]['weights'],list, 
                                      msg='weights data type error')
                if i == 0:
                    self.assertEqual(len(net[i][j]['weights']), n_input+1, 
                                     msg='weight count error')
                else:
                    self.assertEqual(len(net[i][j]['weights']), 
                                     n_hidden_neuron+1, 
                                     msg='weight count error')
                
                
    def AssertListAlmostEqual(self, expected, result, msg):
        self.assertEqual(len(expected), len(result), msg=msg)
        for a, b in zip(expected, result):
             self.assertAlmostEqual(a, b, msg=msg)
    
    
    def test_feedForward(self):
        #classification
        network = [[{'weights': [0.34, 0.12, 0.53]}],
		[{'weights': [0.78, 0.91]}, {'weights': [0.11, 0.31]}]]
        output = feedforward(network,[1, 2],'logistic')
        expected_output = [0.817075904219318, 0.596940729186586]
        self.AssertListAlmostEqual(expected_output, output, 
                                   'feedforward error in classification')
        #regression
        network = [[{'weights': [0.34, 0.12, 0.53]}], 
                   [{'weights': [0.78, 0.91]}]]
        output = feedforward(network,[1, 2],'logistic',regression=True)
        expected_output = 1.496660706922860
        self.assertAlmostEqual(output, expected_output, 
                               msg='feedforward error in regression')
    
    
    def test_backpropagation(self):
        #classification
        network = [[{'weights': [0.34, 0.12, 0.53], 'output': 0.7521291114395702}],
                   [{'weights': [0.78, 0.91], 'output': 0.8170759042193183},
                     {'weights': [0.11, 0.31], 'output': 0.5969407291865862}]]
        backpropagate(network, [0, 1], 'logistic', False)
        delta_1 = network[0][0]['delta']
        delta_2 = network[1][0]['delta']
        delta_3 = network[1][1]['delta']
        self.assertAlmostEqual(delta_1, -0.0157698329885735, 
                               msg='clf:error in computing hidden layer delta')
        self.assertAlmostEqual(delta_2, -0.122122510439718, 
                               msg='clf:error in computing output layer delta')
        self.assertAlmostEqual(delta_3, 0.096977066200573, 
                               msg='clf:error in computing output layer delta')
        #regression
        network = [[{'output': 0.7521291114395702, 
                     'weights': [0.34, 0.12, 0.53]}], 
                    [{'output': 1.4966607069228648, 
                      'weights': [0.78, 0.91]}]]
        backpropagate(network, [0.5], 'logistic', True)
        delta_1 = network[0][0]['delta']
        delta_2 = network[1][0]['delta']
        self.assertAlmostEqual(delta_1, -0.1449305236966680, 
                               msg='rgr:error in computing hidden layer delta')
        self.assertAlmostEqual(delta_2, -0.996660706922865, 
                               msg='rgr:error in computing output layer delta')
        print(network)
        
        
    
    def test_updateWeights(self):
        #calssification
        network = [[{'weights': [0.34, 0.12, 0.53], 
                     'delta': -0.015769832988573537, 
                     'output': 0.7521291114395702}], 
                    [{'weights': [0.78, 0.91], 
                    'delta': -0.12212251043971845, 
                    'output': 0.8170759042193183}, 
                    {'weights': [0.11, 0.31], 
                     'delta': 0.09697706620057299, 
                     'output': 0.5969407291865862}]]
        updateWeights(network,0.01,[1, 2, 1])
        expected_weights = [[0.339842301670114, 0.119684603340229, 0.529842301670114], 
                            [0.779081481047362, 0.908778774895603], 
                            [0.110729392746315, 0.310969770662006]]
        weight_1 = network[0][0]['weights']
        weight_2 = network[1][0]['weights']
        weight_3 = network[1][1]['weights']
        for weight, expected_weight in zip([weight_1, weight_2, weight_3],
                                           expected_weights):
            self.AssertListAlmostEqual(expected_weight, weight, 
                                       'clf: weight update error')
        #regression
        network = [[{'output': 0.7521291114395702, 
                     'delta': -0.14493052369666767, 
                     'weights': [0.34, 0.12, 0.53]}], 
                    [{'output': 1.4966607069228648, 
                      'delta': -0.9966607069228648, 
                      'weights': [0.78, 0.91]}]]
        updateWeights(network, 0.01, [1, 2, 0.5])
        expected_weights = [[0.338550694763033, 0.117101389526067, 0.528550694763033], 
                            [0.772503824680954, 0.900033392930771]]
        weight_1 = network[0][0]['weights']
        weight_2 = network[1][0]['weights']
        for weight, expected_weight in zip([weight_1, weight_2],
                                           expected_weights):
            self.AssertListAlmostEqual(expected_weight, weight, 
                                       'rgr: weight update error')
#        print(network)
        
       
                
if __name__ == "__main__":
    unittest.main()