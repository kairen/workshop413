# coding=utf-8
import sys
import random
import numpy as np

input_dim = 10

# Y =(X1)^2 + (X2)^2+…+(X10)^2
for i in xrange(1000000):
    temp_data = []
    for j in xrange(input_dim):
        r = random.random()
        temp_data.append(r)
    t_data = np.array(temp_data)
    t_sum = np.sum(np.square(t_data))
    temp_data.append(t_sum)
    res_str = ""
    for num in temp_data:
        res_str += str(num) + " "
    print res_str

# python generate_data.py > train-data.txt
