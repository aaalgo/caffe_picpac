import caffe
import random

class PicPacDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        assert len(bottom) == 0
        assert len(top) == 2

    def reshape(self, bottom, top):
        # Copy shape from bottom
        top[0].reshape(1, 3, 256, 256)
        top[1].reshape(1, 1, 256, 256)

    def forward(self, bottom, top):
        # Copy all of the data
        top[0].data[...] = 0
        top[1].data[...] = 0

    def backward(self, top, propagate_down, bottom):
        pass

