This folder contains various scripts used for grid search, model evaluation and image generation. A short description of each script and their usage will be given shortly. Each script is very well documented, and the user is advised to look at the comments inside the scripts for more information about their usage.

- `calculate_accuracy_t.py` - used for calculating IoU, pixel accuracy, recall and precision for the given segmented images
- `colorize.py` - used to transform a semantic map (containing indices of semantic classes) into a *.ppm image
- `copy_files.py` - used in combination with `validation_set_picker.py` to copy appropriate label and potentials files
- `evaluate_grid.py` - used to evaluate grid search results (this was extracted from the actual grid search for usability reasons)
- `grid_config.py` - used to configure grid search parameters
- `grid_search.py` - performs grid search with parameters specified in `grid_config.py`
- `run_crf.py` - used to run the dense crf with a specific set of crf parameters for a set of images
- `validation_set_picker.py` - used for picking a subset of the validation set for faster grid search

## Grid search configuration
In order to configure the grid search parameters, the user needs to modify the `grid_config.py` file. It is a python file and easily readable. It has the following structure:

```python
# Grid search parameters
search_ranges = {
    'smoothness_theta': [3],
    'smoothness_w': [3],
    'appearance_theta_rgb': [3, 4, 5, 6, 7, 8, 9, 10],
    'appearance_theta_pos': [50, 60, 70, 80, 90, 100],
    'appearance_w': [5, 10]
}
```


