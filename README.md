# Piecewise CRF

This is an implementation of piecewise crf training for semantic segmentation based on the work of Chen et al. The implemented model consists of three parts:

1. A neural network used for learning unary and binary potentials
2. A contextual conditional random field that combines the learnt unary and binary potentials
3. A fully connected Gaussian conditional random field used for segmentation postprocessing

The implemented system is evaluated on the publicly available datasets: [Cityscapes](https://www.cityscapes-dataset.com/) and [KITTI](http://adas.cvc.uab.es/s2uad/). For more information about the implementation as well as the results look into the thesis paper.

## Usage
In this section the usage pipeline for semantic segmentation is explained. For more detailed usage explanations about specific scripts look into the comments inside them or the readme files in appropriate subdirectories of this project.
The usage pipeline consists of several steps which will be further explained in the upcoming sections.
All the scripts are well documented and for information about script arguments look into comments.

**IMPORTANT**: In order to run the piecewisecrf scripts, set the PYTHONPATH environment variable to the project(repository) path.

### Generating images
The first step is to generate all the necessary files used for training and validation.

1. Download the datasets ([Cityscapes](https://www.cityscapes-dataset.com/) or [KITTI](http://adas.cvc.uab.es/s2uad/)). For the Cityscapes dataset download the ground truth labels as well as left images. Extract the downloaded archives.
2. Run the `piecewisecrf/datasets/cityscapes/train_validation_split.py` in order to generate the validation dataset. For KITTI use `piecewisecrf/datasets/kitti/train_validation_split.py`.
3. Configure `piecewisecrf/config/prefs.py` file. Set the `dataset_dir, save_dir, img_width, img_height, img_depth` flags
4. Run the `piecewisecrf/datasets/cityscapes/prepare_dataset_files.py` in order to generate files necessary for tensorflow records generation as well as evaluation. For KITTI use `piecewisecrf/datasets/kitti/prepare_dataset_files.py`.
5. Generate tensorflow records used for training and validation by running the following script `piecewisecrf/datasets/prepare_tfrecords.py`. The destination directory is used to reconfigure the `piecewisecrf/config/prefs.py` file (`train_records_dir, val_records_dir, test_records_dir` flags)

### Training the neural network

1. Prepare the numpy file with vgg weights (look at the readme in caffe-tensorflow).
2. Configure the `piecewisecrf/config/prefs.py` (`vgg_init_file, train_dir` and all the other parameters for training)
3. Run `piecewisecrf/train.py`

### Evaluating the neural network
1. Configure the `piecewisecrf/config/prefs.py` if not already done.
2. Run the following script: `piecewisecrf/eval.py`

### Generating output files from contextual CRF
1. Configure the `piecewisecrf/config/prefs.py` if not already done.
2. Run the following script: `piecewisecrf/forward_pass.py`

This will generate predictions (in small and original resolution) as well as unary potentials used by the fully connected CRF.

### Learning the parameters of the fully connected CRF
This is done by applying grid search.

0. Build the dense crf executable (look at the readme in densecrf)
1. If necessary pick a subset of the validation dataset by using `tools/validation_set_picker.py` and `tools/copy_files.py`
2. Configure the `tools/grid_config.py` file (grid search parameters)
3. Start the grid search by running `tools/grid_search.py`.
4. Evaluate grid search results by running `tools/evaluate_grid.py`

With this you will get optimal CRF parameters on the validation dataset.

### Fully connected CRF inference and evaluation

1. To infer images with the fully connected CRF run the `tools/run_crf.py` script.
2. In order to evaluate the generated output you can use `tools/calculate_accuracy_t.py`
3. Because the output is in binary format, in order to generate image files, run the `tools/colorize.py` script.

## References

> Efficient Piecewise Training of Deep Structured Models for Semantic Segmentation <br/>
> Guosheng Lin,  Chunhua Shen, Anton van den Hengel, Ian Reid <br/>
> IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016 <br/>

> Convolutional scale invariance for semantic segmentation <br/>
> Krešo Ivan, Čaušević Denis, Krapac Josip, Šegvić Siniša <br/>
> 38th German Conference on Pattern Recognition, Hannover, 2016 <br/>

> Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials <br/>
> Philipp Krähenbühl and Vladlen Koltun <br/>
> NIPS 2011 <br/>

> Vision-based offline-online perception paradigm for autonomous driving <br/>
> Ros, G., Ramos, S., Granados, M., Bakhtiary, A., Vazquez, D., Lopez, A.M. <br/> 
> IEEE Winter Conference on Applications of Computer Vision, Hawaii, 2015 <br/>

> The Cityscapes dataset for semantic urban scene understanding <br/>
> Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, SS., Schiele, B. <br/>
> Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, Las Vegas, 2016 <br/>
