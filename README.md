# Pombe Phenosizer

A rapid machine learning-based method to measure cell dimensions in S. pombe.

## Description

We have developed a streamline approach to acquiring cell dimensions in S. pombe. With this pipeline, one could go from microscopy images to statistical results about ten times faster than manual segmentation and measurments. It is built on the PyTorch framework with the utilization of DeeplabV3 for training purposes.

This program takes a list of microscopy images of fission yeast cells as its input and it will output a csv file containing the count of cells, x and y pixel coordinates of the cell, and its cell dimensions (area, length, and width).

The process is accomplished by utilizing a neural network to create an initial mask image of the cells that it was able to identify and then thresholding was applied to clean up the initial mask. After, each cell was filled with a different grayscale color and the dimensions were measured using opencv functions. The program can optionally save any of the intermediate images throughout the process by using optional arguments.

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
3. Run the below command if you have two images (image1.tif and image2.tif) and a weights file called 'weights.pt', which is a trained PyTorch model. If the weights file is called 'weights.pt' you do not need to add the optional argument, [--weights_file <filename>]. This weights argument is only for when you have a specific weights file that you want to use. The below command will produce image1.csv and image2.csv:
```
python process_images.py --weights_file weights.pt image1.tif image2.tif
```
* You can add optional parameters to save the intermediate pictures with these parameters:
```
python process_images.py [--weights_file <filename>] [--write_nn_mask] [--write_threshold_mask] [--write_filled_cells]
```
* If you have multiple images you can use the \*.tif shortcut to loop through all of the images
```
python process_images.py *.tif
```

#### WindowsOS instructions:

* Same starting process as MacOS instructions except use this command:
```
for %i in (*.tif) do python3 process_images.py %i [--write_nn_mask] [--write_threshold_mask] [--write_filled_cells] [--weights_file <filename>]
```
## Help

Running the program without any arguments like so:
```
python process_images.py *.tif
```
will bring the help menu up for more information on what arguments can be passed through
## Authors

Contributors names and contact info

* Martin Vo(vom@xavier.edu)
* Nathan Sommer(sommern1@xavier.edu)
* Ryan Miller(millerr33@xavier.edu)

## Version History

* 1.0
    * Initial Release

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE.md)

## Acknowledgments

Inspiration, code snippets, etc.
* [Training code](https://github.com/msminhas93/DeepLabv3FineTuning)
