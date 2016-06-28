# Fully connected CRF

This is a modified version of the fully connected CRF code from Krähenbühl and Koltun:

> Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials <br/>
> Philipp Krähenbühl and Vladlen Koltun <br/>
> NIPS 2011 <br/>

## Files
The `dense_inference.cpp` file contains the main program for starting inference. All the other files contain the required logic for inference and crf representation and will not be further explained. For more information look inside the file in question.

## Build
In order to build the provided code, position yourself inside the parent directory (root directory of the project/git repo) and type:

```
make all
```

This will create a DenseCRFv1.out executable which can be used for running CRF inference

## Usage
The DenseCRFv1.out file is used for CRF inference. Example usage:

```
DenseCRFv1.out dataset input_image compressed_unary output_img sigmaSmoothness weightSmoothness positionSigmaBi colorSigmaBi weightBi
```

The arguments are as follows:

- `dataset` - kitti or cityscapes
- `input_image` - *.ppm input image to be segmented
- `compressed_unary` - *.bin unary potentials
- `output_img` - destination file
- `sigmaSmoothness` - smoothness kernel parameter
- `weightSmoothness` - smoothness kernel weight
- `positionSigmaBi` - appearance kernel parameter (for pixel positions)
- `colorSigmaBi` - appearance kernel parameter (for pixel colors)
- `weightBi` - appearance kernel weight
