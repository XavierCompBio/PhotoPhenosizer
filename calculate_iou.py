import cv2
import sys
import numpy as np


def calculate_iou(image1, image2):
    
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # everywhere that is not a white cell, change to 1
    image1[image1 != 255] = 1
    image2[image2 != 255] = 2


    cell_color = 255
    AOU_Total = 0
    AOI_Total = 0
    # anywhere where the pixels are the same
    AOI = np.count_nonzero(image1 == image2)
    
    image1_cell_area = np.count_nonzero(image1 == cell_color)
    
    image2_cell_area = np.count_nonzero(image2 == cell_color)

    AOU = (image1_cell_area + image2_cell_area) - AOI

    print(AOI/AOU)


def main():

    if len(sys.argv) != 3:
        sys.exit("You must provide two masks to compare.")
    
    image1 = cv2.imread(sys.argv[1])
    image2 = cv2.imread(sys.argv[2])

    calculate_iou(image1, image2)


if __name__ == "__main__":
    main()
    


