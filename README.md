# Pombe Phenosizer

A rapid machine learning-based method to measure cell dimensions in S. pombe.

## Description

We have developed a streamline approach to acquiring cell dimensions in S. pombe. With this pipeline, one could go from microscopy images to statistical results about ten times faster than manual segmentation and measurments. It is built on the PyTorch framework with the utilization of DeeplabV3 for training purposes.

## Getting Started

### Dependencies

* Libraries that are required to run the program are: numpy, torch, torchvision, pillow, opencv-python, matplotlib. See requirements.txt to know which libraries to install.
* PombePhenosizer (PP) is able to operate on MacOS, WindowsOS, and Linux.

### Installing

* Set up a virtual environment and install all necessary libraries from requirements.txt
* [creating a virtual environment in Windows](docs/windows_venv.md)
* [creating a virtual environment in MacOS](docs/macos_venv.md)
* Training pipeline of the PP program came from this repository: https://github.com/msminhas93/DeepLabv3FineTuning 

### Executing program
#### MacOS instructions:

* First make sure that a virtual environment is setup and contains all of the libraries
1. First, put all of the microscopy images into a folder and change the directory path in the terminal to the path where the images are
2. Put the trained weights file into the same folder as this repository. To get a sample trained weights file, contact the authors. For training your own weights file from your own segmented images: [Training code](https://github.com/msminhas93/DeepLabv3FineTuning)
3. Run the below command:
```
python process_images.py *.tif --write_nn_mask --write_threshold_mask --write_filled_cells --weights_file [name of weights file here]
```
* If don't want to create all of the intermediate files such as the mask or filled_cells images, then one can remove those parameters:
```
python process_images.py *.tif --weights_file [name of weights file here]
```
* If weights file name is just "weights.pt", then one can remove that parameter:
```
python process_images.py *.tif
```

#### WindowsOS instructions:

* Same starting process as MacOS instructions except use this command:
```
for %i in (*.tif) do python3 process_images.py %i [--write_nn_mask] [--write_threshold_mask] [--write_filled_cells] [--weights_file [name of file]]
```
## Help


## Authors

Contributors names and contact info

* Martin Vo(vom@xavier.edu)
* Nathan Sommer(sommern1@xavier.edu)
* Ryan Miller(millerr33@xavier.edu)

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [Training code](https://github.com/msminhas93/DeepLabv3FineTuning)
