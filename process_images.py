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


@dataclass
class Dimensions:
    lengthCircle: float
    lengthEllipse: float
    widthEllipse: float
    areaContour: float


def write_image(original_filename, string_label, image):
    original_path = Path(original_filename)
    new_stem = original_path.stem + string_label
    new_filename = original_path.with_stem(new_stem)
    cv2.imwrite(str(new_filename), image)


def process_image(image_filename, args):
    """
    Processes the (image_filename) image to get dimensions of areas, lengths, and width in pixels

    :param image_filename: the name of the image
    :return: the csv file of all the metrics in a table
    """
    input_img = Image.open(image_filename).convert("RGB")
    nn_mask = nn_predict(input_img, args.weights_file)
    threshold_mask = threshold(nn_mask)
    blob_keypoints = detect_blobs(threshold_mask)
    filled_cells = fill_cells(threshold_mask, blob_keypoints)
    areas = calculate_areas(filled_cells, blob_keypoints)
    dimensions = calculate_dimensions(filled_cells, len(areas))
    write_result(image_filename, areas, dimensions, blob_keypoints)
    if args.write_nn_mask:
        write_image(image_filename, '-nn_mask', nn_mask)
    if args.write_threshold_mask:
        write_image(image_filename, '-threshold', threshold_mask)
    if args.write_filled_cells:
        write_image(image_filename, '-filled_cells', filled_cells)


def nn_predict(input_img, weights_filename):
    """
    Uses a trained NN model to segment the cells from the image

    :param input_img: the input image
    :return: the image of the cells from the NN
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
    # out = stores output.tif image

    with TemporaryDirectory() as temp_dir_path:
        temp_file_path = os.path.join(temp_dir_path, 'output.tif')
        save_image(out, 'output.tif')
        save_image(out, temp_file_path)
        out = cv2.imread(temp_file_path, cv2.IMREAD_GRAYSCALE)

    return out


def threshold(nn_mask):
    """
    Thresholds the NN image

    :param nn_mask: the NN image
    :return: the threshold mask of the NN image
    """
    th, threshold_mask = cv2.threshold(nn_mask, 200, 255, cv2.THRESH_BINARY)
    return threshold_mask


def detect_blobs(threshold_mask):
    """

    """
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


def calculate_areas(filled_cells, blob_keypoints):
    cell_count = len(blob_keypoints)

    # calculate the areas
    areas = cv2.calcHist([filled_cells], [0], None, [256], [0, 255])
    # slice the array down based on the number of cells
    areas = areas[1:cell_count + 1, 0]

    return areas


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


def main():
    parser = ArgumentParser()
    parser.add_argument('--write_nn_mask', action='store_true',
                        help='Write the mask images produced by the neural '
                             'network')
    parser.add_argument('--write_threshold_mask', action='store_true',
                        help='Write the result of thresholding')
    parser.add_argument('--write_filled_cells', action='store_true',
                        help='Write the result of filling detected cells with '
                             'different grays')
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
