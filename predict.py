#!/usr/bin/env python3
import numpy as np
import cv2
import caffe

WIDTH = 640
HEIGHT = 400

net = caffe.Net('deploy.prototxt', 'snapshots/x_iter_1000.caffemodel', caffe.TEST)

image = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
image = cv2.resize(image, (WIDTH,HEIGHT))
# caffe dimension order is NCHW
batch = np.transpose(image[np.newaxis, :, :, :], [0, 3, 1, 2])
net.blobs['data'].data[...] = batch
out = net.forward()
prob = out['prob'][0, 1, :, :]  # channel 1 is probability
cv2.imwrite('prob.jpg', prob * 255)


