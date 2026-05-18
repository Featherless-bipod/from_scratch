import numpy as np 
import pandas as pd 
import nMLP as nMLP
import matplotlib.pyplot as plt
import time
import test as t

route = '../data/TMNIST_data.csv'
#route = 
testAmount = 1000
scale = 255
results = 10
rem_axis = "names"

#insert model paramters
hidden_layers = 2
nodes_per_layer = 10
learning_rate = 0.1
epochs = 100

test_y,test_x,train_y,train_x, pixels,train_trials,results = nMLP.process_data(route, testAmount, scale, results,rem_axis=rem_axis)
params = nMLP.gradient_descent(hidden_layers,nodes_per_layer,train_x,train_y,pixels,results,epochs,learning_rate,schedule="fixed",batch_size = 64)

