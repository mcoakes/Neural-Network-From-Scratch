import numpy as np
from collections import deque

def sigmoid(x, deriv = False):
    return 1 / 1 + (np.exp(-x)) if not deriv else sigmoid(x) * (1 - sigmoid(x))

class NNetwork:
    def __init__(self,inputs,num_hidden,hidden_shapes,out_shape,
                     activate=sigmoid):
        self.activate = activate
        self.input = inputs
        self.in_shape = inputs.shape
        self.num_hidden = num_hidden
        self.hidden_shapes = hidden_shapes
        self.out_shape = out_shape
        
        self.layers = []
        self.hidden_shapes = deque(self.hidden_shapes)
        hid_shape = self.hidden_shapes.popleft() \
                        if self.hidden_shapes else out_shape 
        hid_shape2 = self.hidden_shapes.popleft() \
                         if self.hidden_shapes else hid_shape
        self.in_layer = Layer(self.input,hid_shape)
        self.layers.append(self.in_layer)
        self.hidden_layers = []

        for _ in range(self.num_hidden - 1):
            self.hidden_layers.append(Layer(np.empty(hid_shape),hid_shape2))
            hid_shape = hid_shape2
            hid_shape2 = self.hidden_shapes.popleft() \
                             if self.hidden_shapes else hid_shape

        if self.num_hidden > 1:
            self.hidden_layers.append(Layer(np.empty(hid_shape2),
                                                self.out_shape))
        self.layers += self.hidden_layers
        self.out_layer = Layer(np.empty(self.layers[-1].out_shape))
        self.layers.append(self.out_layer)

    def feed_forward(self):
        out = self.input
        for l in self.layers:
            l.input = out
            if l.out_shape[0]:
               out = self.activate(np.dot(l.input,l.weights) + l.biases)
               l.output = out

class Layer:
    def __init__(self,inputs,out_shape=((),)):
        self.input = np.array(inputs)
        self.in_shape = self.input.shape
        self.output = None
        self.out_shape = out_shape
        self.shape = (*self.in_shape,*self.out_shape) \
                         if self.out_shape[0] else self.in_shape
        self.weights = np.random.random((*self.input.shape,*self.out_shape)) \
                           if self.out_shape[0] else None
        self.biases = np.random.random((*self.out_shape,)) \
                          if self.out_shape[0] else None
