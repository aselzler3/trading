# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 17:34:55 2018

@author: Andrew Selzler
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Network(object):
    def __init__(self, sizes):
        self.num_layers=len(sizes)
        self.sizes=sizes
        self.biases=[np.random.randn(y,1) for y in sizes[1:]]
        self.weights=[np.random.rand(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
    
    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            a=sigmoid(np.dot(w,a)+b)
        return a
        
    def SGD(self, data, eta):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        result=[]
        n=len(data)
        for x,y in data:
            result.append(self.feedforward(x))
            self.update_mini_batch([(x,y)],eta)
        return result
    def update_mini_batch(self,mini_batch,eta):
         """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate."""
         nabla_b=[np.zeros(b.shape) for b in self.biases] 
         nabla_w=[np.zeros(w.shape) for w in self.weights]
         for x,y in mini_batch:
             delta_nabla_b,delta_nabla_w=self.backprop(x,y)
             nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
             nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
         self.weights=[w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
         self.biases=[b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)]
                
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
        
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
        
def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))
        
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
    
def make_dataset(arr,n_days):
    result=[]
    for i in range(n_days,len(arr)):
        result.append((np.array(arr[i-n_days:i]).reshape(-1,1), arr[i]))
    return result
def process_data(arr):
    result=[]
    for i in range(1,len(arr)):
        result.append(sigmoid(arr[i]-arr[i-1]))
    return result
    
def R_squared(y,f):
    # y and f are arrays
    SSres=0
    SStot=0
    m=np.mean(y)
    for i in range(len(y)):
        SSres+=(y[i]-f[i])**2
        SStot+=(y[i]-m)**2
    return 1-SSres/SStot
    
name_e='C:/Users/Andrew Selzler/Desktop/CHF JPY Historical Data (early).csv'
name_l='C:/Users/Andrew Selzler/Desktop/CHF JPY Historical Data (late).csv'
df_e=pd.read_csv(name_e)
df_e=df_e.drop([len(df_e)-2,len(df_e)-1],axis=0)
df_e=df_e.iloc[::-1].reset_index()
df_e['Price']=df_e['Price'].apply(float)

df_l=pd.read_csv(name_l)
df_l=df_l.drop([len(df_l)-2,len(df_l)-1],axis=0)
df_l=df_l.iloc[::-1].reset_index()
df_l['Price']=df_l['Price'].apply(float)

P=np.concatenate((np.array(df_e['Price']),np.array(df_l['Price'])),axis=0)
n_days=10
data=make_dataset(process_data(P),n_days)
model=Network([n_days,10,1])
predictions=model.SGD(data,eta=9)
predictions=[predictions[i][0][0] for i in range(len(predictions))]
Y=[x[1] for x in data]
diff=[Y[i]-predictions[i] for i in range(len(Y))]
plt.scatter([i for i in range(len(diff))],diff)
plt.show()

moving_R=[]
for i in range(10,len(Y)):
    moving_R.append(R_squared(np.array(Y[i-10:i]),np.array(predictions[i-10:i])))
plt.plot(moving_R[600:2000])
plt.show()