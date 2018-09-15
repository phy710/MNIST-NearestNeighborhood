# -*- coding: utf-8 -*-
"""
Load MNIST Dataset

Created on Sat Apr 28 14:00:27 2018

@author: Zephyr
"""

import numpy
import struct

def loadMNISTLabels(filename):
    bin_data = open(filename, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = numpy.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

def loadMNISTImages(filename):
    bin_data = open(filename, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = numpy.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        images[i] = numpy.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images