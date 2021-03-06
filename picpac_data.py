import os
import caffe
import random
import picpac

WIDTH = 640
HEIGHT = 400
BATCH = 16

def create_picpac_stream (path, is_training):
    assert os.path.exists(path)
    config = {"db": path,
              "loop": is_training,
              "shuffle": is_training,
              "reshuffle": is_training,
              "annotate": [1],
              "channels": 3,
              "stratify": is_training,
              "dtype": "float32",
              "batch": BATCH,
              "colorspace": 'BGR',
              "cache": 0,
              "order": "NCHW",  # specific to caffe
              "transforms": [
                  {"type": "resize", "width": WIDTH, "height": HEIGHT},
                  {"type": "augment.flip", "horizontal": True, "vertical": False, "transpose": False},
                  {"type": "augment.scale", "min":0.8, "max":1.2},
                  {"type": "augment.rotate", "range": 10},
                  {"type": "augment.add", "range":20},
                  {"type": "clip", "width": WIDTH, "height": HEIGHT},
                  {"type": "rasterize"},
                  ]
             }

    if not is_training:
        config['threads'] = 1
    return picpac.ImageStream(config)

class PicPacDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        self.stream = create_picpac_stream(self.param_str, True)
        assert len(bottom) == 0
        assert len(top) == 2
        pass

    def reshape(self, bottom, top):
        top[0].reshape(BATCH, 3, HEIGHT, WIDTH)
        top[1].reshape(BATCH, HEIGHT, WIDTH)
        pass

    def forward(self, bottom, top):
        meta, images, labels = self.stream.next()
        top[0].data[...] = images
        top[1].data[...] = labels[:, 0, :, :]
        pass

    def backward(self, top, propagate_down, bottom):
        pass

