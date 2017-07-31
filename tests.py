#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:31:18 2017

@author: kasperipalkama
"""

from feedforward_multilayer_perceptron import initialize, feedforward
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
                
    
    def test_feedForward(self):
        network = [[{'weights': [0.34, 0.12, 0.53]}],
		[{'weights': [0.78, 0.91]}, {'weights': [0.11, 0.31]}]]
        #classification
        output = feedforward(network,[1, 2],'logistic')
        expected_output = [0.817075904219318, 0.596940729186586]
        self.assertEqual(len(expected_output), len(output))
        for a, b in zip(expected_output, output):
             self.assertAlmostEqual(a, b, msg='feedforward error in '\
                                    'classification')
        #regression
        network = [[{'weights': [0.34, 0.12, 0.53]}],
		[{'weights': [0.78, 0.91]}, {'weights': [0.11, 0.31]}],
        [{'weights': [0.87, 0.64, 0.11]}]]
        output = feedforward(network,[1, 2],'logistic',regression=True)
        expected_output = 1.20289810335022
        self.assertAlmostEqual(output, expected_output, 
                               msg='feedforward error in regression')
                
    

def main():
    unittest.main()
    
if __name__ == "__main__":
    main()