# Caffe to TensorFlow

This is the publicly available converter [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) used for converting [Caffe](https://github.com/BVLC/caffe/) models to [TensorFlow](https://github.com/tensorflow/tensorflow).

# Usage

Run `convert.py` to convert an existing Caffe model to TensorFlow.

Make sure you're using the latest Caffe format.

The output consists of two files:

1. A data file (in NumPy's native format) containing the model's learned parameters. (This is used in piecewisecrf training)
2. A Python class that constructs the model's graph.

# Converting VGG16

1. Download the VGG16 Caffe model [here](https://gist.github.com/jimmie33/27c1c0a7736ba66c2395)
2. Convert the model to the new Caffe format by using `upgrade_net_proto_text` and `upgrade_net_proto_binary` tools that ship with Caffe. Also make sure you're using a fairly recent version of Caffe.
3. Run `convert.py` and save the `.npy` in an easily rememberable place. It will be used for initializing the weights in the potentials generator network in piecewisecrf.