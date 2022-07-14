import csv
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage import morphology
from IPython.display import display

@dataclass
class Dimensions:
    lengthCircle: float
    lengthEllipse: float
    widthEllipse: float
    areaContour: float


def write_image(original_filename, string_label, image):
    """
    Writes an image to a file. The filename is constructed by inserting string_label
    before the extension of original_filename.

    For example: if original_filename is 'image.tif' and string_label is '-label', 
    then the filename of the new file will be 'image-label.tif'

    :param original_filename: the name of the image
    :param string_label: the name of the arguments
    :param image: the wanted image based on the arguments
    :return: the images at the argument checkpoints
    """
    original_path = Path(original_filename)
    new_stem = original_path.stem + string_label
    new_filename = original_path.with_stem(new_stem)
    cv2.imwrite(str(new_filename), image)


def process_image(image_filename, args):
    """
    Processes the (image_filename) image to get dimensions of 
    areas, lengths, and width in pixels. Writes out a csv file 
    with the number of cells and its dimensions

    :param image_filename: the name of the image
    :param args: any arguments that was passed from the user
    to the terminal
    """
    input_img = Image.open(image_filename).convert("RGB")
    nn_mask = nn_predict(input_img, args.weights_file)
    threshold_mask = threshold(nn_mask)
    threshold_mask = erod_dilate(threshold_mask)
    area_filtered, label_img_area = area_filter(threshold_mask)
    # Comment iou function out if no need for it
    # calculate_iou(label_img_area)
    feret(area_filtered, image_filename)
    if args.write_nn_mask:
        write_image(image_filename, '-nn_mask', nn_mask)
    if args.write_threshold_mask:
        write_image(image_filename, '-threshold', threshold_mask)
    if args.write_area_filter:
        write_image(image_filename, '-area_filter', label_img_area)


def nn_predict(input_img, weights_filename):
    """
    Uses a trained NN model to segment the cells from the image

    :param input_img: the input image
    :param weights_filename: if an argument was passed through to
    specify another weights file to be used, then use it
    :return: an approximation of a mask indicating where the cells are
    """
    model = torch.load(weights_filename)
    model.eval()
    model.to('cpu')

    data_transforms = transforms.Compose([transforms.ToTensor()])
    # transforms in torchvision only work on PIL images?
    input_img = data_transforms(input_img)

    # nn.Conv2d(?) expects a 4-dimensional input tensor as [batch_size, channels, height, width], so unsqueeze is necessary
    input_img = input_img.unsqueeze(0)
    input_img.to('cpu')

    torch.set_grad_enabled(False)
    prediction = model(input_img)
    out = prediction['out'].data.cpu()

    # saves output image in a temporary directory to be later deleted
    with TemporaryDirectory() as temp_dir_path:
        temp_file_path = os.path.join(temp_dir_path, 'output.tif')
        save_image(out, temp_file_path)
        out = cv2.imread(temp_file_path, cv2.IMREAD_GRAYSCALE)

    return out


def threshold(nn_mask):
    """
    Thresholds the NN image

    :param nn_mask: the NN image
    :return: the threshold mask of the NN image
    """
    # any pixels that are above the first number, turn it into 255 (white)
    th, threshold_mask = cv2.threshold(nn_mask, 170, 255, cv2.THRESH_BINARY)
    return threshold_mask

def erod_dilate(threshold_mask):
    """
    Applies closing, erosion, and dilation morphological changes to the threshold image 
    to close any holes, remove noise, and separate groups of cells.

    :param threshold_mask: threshold mask of the NN image
    :return: another threshold mask that have the new morphological changes
    """
    # closing
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(
        threshold_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # erosion
    kernel = np.ones((3, 3), np.uint8)
    thresh_erosion = cv2.erode(closing, kernel, iterations=3)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(
        thresh_erosion, cv2.MORPH_OPEN, kernel, iterations=4)

    # dilation
    kernel = np.ones((3, 3), np.uint8)
    thresh_dilation = cv2.dilate(opening, kernel, iterations=2)

    return thresh_dilation

def area_filter(threshold_mask):

    image = threshold_mask.copy()
    arr = image > 0
    area_filtered = morphology.remove_small_objects(arr, min_size=700)
    
    label_img = label(area_filtered)
    label_img_area = label_img.astype(np.uint8)
    label_img_area[label_img_area != 0] = 255
    return area_filtered, label_img_area

def calculate_iou(label_img_area):
    pp_image = label_img_area.copy()
    manual = cv2.imread('NT_x40_5_mask.tif')
    manual = cv2.cvtColor(manual, cv2.COLOR_BGR2GRAY)
    manual_copy = manual.copy()
    pp_copy = pp_image.copy()

    # everywhere that is not a white cell, change to 1
    manual_AOI = manual_copy.copy()
    pp_AOI = pp_copy.copy()
    manual_AOI[manual_AOI != 255] = 1
    pp_AOI[pp_AOI != 255] = 2


    cell_color = 255
    AOU_Total = 0
    AOI_Total = 0
    # anywhere where the pixels are the same
    AOI = np.count_nonzero(manual_AOI == pp_AOI)
    
    manual_cell_area = np.count_nonzero(manual_copy == cell_color)
    
    pp_cell_area = np.count_nonzero(pp_copy == cell_color)

    AOU = (manual_cell_area + pp_cell_area) - AOI

    print(AOI/AOU)


def feret(area_filtered, image_filename):

    csv_filename = Path(image_filename).with_suffix('.csv')
    
    image = area_filtered.copy()

    label_img = label(image)

    props = regionprops_table(label_img, properties=(
                                                    'area_filled',
                                                    'feret_diameter_max',
                                                    'axis_minor_length'))

    df = pd.DataFrame(props)

    df.to_csv(str(csv_filename))

def main():
    """
    Main function with listed arguments that can be passed through
    the terminal
    """
    parser = ArgumentParser()
    parser.add_argument('--write_nn_mask', action='store_true',
                        help='Write the mask images produced by the neural '
                             'network')
    parser.add_argument('--write_threshold_mask', action='store_true',
                        help='Write the result of thresholding')
    parser.add_argument('--write_area_filter', action='store_true',
                        help='Write the result of area filtering')
    parser.add_argument('--weights_file',
                        help='Specify the path to the weights file')
    parser.add_argument('image_files', nargs='+')
    args = parser.parse_args()
    if args.weights_file is None:
        args.weights_file = 'weights.pt'
    for filename in args.image_files:
        process_image(filename, args)


if __name__ == "__main__":
    main()
