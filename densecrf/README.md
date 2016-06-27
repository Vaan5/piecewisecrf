# Fully connected CRF

This is a modified version of the fully connected CRF code from Krähenbühl and Koltun:

> Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials
> Philipp Krähenbühl and Vladlen Koltun
> NIPS 2011

## Build
In order to build the provided code, position yourself inside the parent directory (root directory of the project/git repo) and type:
```
make all
```


## Usage

The dense_inference.cpp file is used for CRF inference. Example usage:
"Usage: %s dataset input_image compressed_unary output_img sigmaSmoothness weightSmoothness positionSigmaBi colorSigmaBi weightBi\n

