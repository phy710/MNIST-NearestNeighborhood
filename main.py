# -*- coding: utf-8 -*-
"""
Nearest Neighborhood for MNIST Dataset

Created on Sat Apr 28 14:07:39 2018

@author: Zephyr
"""

import numpy
import loadMNIST
import time

time_start=time.time()
trainData = loadMNIST.loadMNISTImages('train-images.idx3-ubyte')
trainLabels = loadMNIST.loadMNISTLabels('train-labels.idx1-ubyte')
testData = loadMNIST.loadMNISTImages('t10k-images.idx3-ubyte')
testLabels = loadMNIST.loadMNISTLabels('t10k-labels.idx1-ubyte')
estimatedLabels = numpy.zeros(10000)
for a in range(10000):
    diff = testData[a, :, :] - trainData[0, :, :]
    diff2 = numpy.dot(diff.reshape(1, 784), diff.reshape(784, 1))
    diff2Min = diff2
    estimatedLabels[a] = trainLabels[1]
    for b in range(1, 60000):
        diff = testData[a, :, :] - trainData[b, :, :]
        diff2 = numpy.dot(diff.reshape(1, 784), diff.reshape(784, 1))
        if diff2 < diff2Min:
            diff2Min = diff2
            estimatedLabels[a] = trainLabels[b]
    print('Calculating...', (a+1)/100, '%')
accuracy = numpy.sum(estimatedLabels == testLabels) / 10000
time_end=time.time()
print('Elapsed time: ', time_end-time_start, 's')
print('Test data accuracy: ', accuracy*100, '%')