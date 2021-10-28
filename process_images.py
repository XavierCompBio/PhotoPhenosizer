import sys
import csv
from tempfile import TemporaryDirectory
import os
from pathlib import Path
# Imports for using model to predict microscope image
import numpy
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
# Imports for thresholding, blob, and area
import cv2
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------
from dataclasses import dataclass


@dataclass
class Dimensions:
    lengthCircle: float
    lengthEllipse: float
    widthEllipse: float
    areaContour: float




def process_image(image_filename):
    """
    
    """
    input_img = Image.open(image_filename).convert("RGB")
    nn_mask = nn_predict(input_img)
    threshold_mask = threshold(nn_mask)
    blob_keypoints = detect_blobs(threshold_mask)
    filled_cells = fill_cells(threshold_mask, blob_keypoints)
    areas = calculate_areas(filled_cells, blob_keypoints)
    dimensions = calculate_dimensions(filled_cells, len(areas))
    write_result(image_filename, areas, dimensions, blob_keypoints)

# ----------------------------------------------------------------------------------------
# code for model to predict microscope image
# returns nn_mask = out


def nn_predict(input_img):
    model = torch.load('weights.pt')
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
    # out = stores output.tif image

    with TemporaryDirectory() as temp_dir_path:
        temp_file_path = os.path.join(temp_dir_path, 'output.tif')
        save_image(out, 'output.tif')
        save_image(out, temp_file_path)
        out = cv2.imread(temp_file_path, cv2.IMREAD_GRAYSCALE)
        

    return out

# ----------------------------------------------------------------------------------------
# threshold code
# returns threshold_mask = im

def threshold(nn_mask):
    # nn_mask - predicted img
    # thresholds (binary) image; any pixels above 215 gets turned into 255 (white)
    

    th, threshold_mask = cv2.threshold(nn_mask, 200, 255, cv2.THRESH_BINARY)
    
    return threshold_mask

# ----------------------------------------------------------------------------------------
# blob detection code
# returns blob_keypoints = keypoints


def detect_blobs(threshold_mask):
    # invert threshold im
    # threshold - inverted threshold img (black -> white; white -> black)
    # -> white cells are now black and black background is now white
    retval, threshold = cv2.threshold(
        threshold_mask, 200, 255, cv2.THRESH_BINARY_INV)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 21

    # Filter by Area.
    # maxArea = 2685 when threshold = 200
    # maxArea = 2534 when threshold = 210
    # maxArea = 2451 when threshold = 215
    params.filterByArea = True
    params.minArea = 500
    params.maxArea = 2685

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.2

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    blob_keypoints = detector.detect(threshold)

    return blob_keypoints

# ----------------------------------------------------------------------------------------
# filled image function
# returns filled_img
def fill_cells(threshold_mask, blob_keypoints):
    im_floodfill = threshold_mask.copy()

    if len(blob_keypoints) > 254:
        print("Warning: Image contains more than 254 cells. Only the first 254 cells will be included.")

    cell_color = 1
    for keypoint in blob_keypoints[:254]:
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        cv2.floodFill(im_floodfill, None, (x, y), cell_color)
        cell_color += 1

    # change white pixels to black
    im_floodfill[im_floodfill == 255] = 0

    return im_floodfill

# ----------------------------------------------------------------------------------------
# areas function
def calculate_areas(filled_cells, blob_keypoints):
 
    cell_count = len(blob_keypoints)

    # calculate the areas
    areas = cv2.calcHist([filled_cells], [0], None, [256], [0, 255])
    # slice the array down based on the number of cells
    areas = areas[1:cell_count+1, 0]

    return areas

# ----------------------------------------------------------------------------------------
# dimensions function
# results - measure dimensions of each cell via contour
def calculate_dimensions(filled_img, cell_count):
    dimensions = []
    for color in range(1, cell_count + 1):
        isolated_img = filled_img.copy()
        # change all pixels that do not belong to this cell to black, so isolated_img
        # is all white except for the cell we want
        isolated_img[filled_img != color] = 0
        isolated_img[filled_img == color] = 255
        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        cnts, hierarchy = cv2.findContours(
            isolated_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        isolated_img = cv2.drawContours(
            isolated_img, cnts, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        ellipse = cv2.fitEllipse(cnts[0])
        lengthEllipse = max(ellipse[1])
        widthEllipse = min(ellipse[1])

        (x, y), r = cv2.minEnclosingCircle(cnts[0])
        lengthCircle = r * 2

        areaContour = cv2.contourArea(cnts[0])
        dimensions.append(Dimensions(
            lengthCircle, lengthEllipse, widthEllipse, areaContour))

    return dimensions

# ----------------------------------------------------------------------------------------
# results function
#


def write_result(image_filename, areas, dimensions, blob_keypoints):
    csv_filename = Path(image_filename).with_suffix('.csv')
    with open(csv_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'x', 'y', 'area', 'areaContour', 'lengthCircle',
                         'lengthEllipse', 'widthEllipse'])
        for cell_id, (area, keypoint, dimension) in enumerate(zip(areas, blob_keypoints, dimensions), start=1):
            writer.writerow([cell_id, int(keypoint.pt[0]), int(keypoint.pt[1]), area, dimension.areaContour,
                             dimension.lengthCircle, dimension.lengthEllipse,
                             dimension.widthEllipse])

# ----------------------------------------------------------------------------------------
# master function


def main():
    for image_filename in sys.argv[1:]:
        process_image(image_filename)


if __name__ == "__main__":
    main()
