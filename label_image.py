import sys
import cv2
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from random import shuffle
from pathlib import Path
import csv
import os
import feret

def label_image(predicted_mask, original_image, original_filename): 

    original_path = Path(original_filename)
    new_stem = original_path.stem + '-labeled'
    original_labeled_filename = original_path.with_stem(new_stem)

    csv_filename = Path(original_labeled_filename).with_suffix('.csv')

    label_predicted = label(predicted_mask)
    label_predicted_mask = label_predicted.copy()
    label_predicted_mask[label_predicted_mask != 0] = 255
    cell_count = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    regions = regionprops(label_predicted)
    shuffle(regions)
    
    for region in regions:
        x, y = region.centroid
        x = int(x)
        y = int(y)

        
        cv2.putText(original_image, str(cell_count), (y, x), font, 1, (255, 0, 0), 2)
        cv2.circle(original_image, (y, x), 0, (0, 255, 0), 10)
        cell_count += 1

    cv2.imwrite(str(Path('labeled_images') / original_labeled_filename), original_image)
    with open(str(Path('labeled_csv') / csv_filename), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Number', 'Area', 'Feret', 'MinFeret'])
        cell_count = 1
        for region in regions:
            label_img_copy = label_predicted.copy()
            label_img_copy[label_img_copy != region.label] = 0

            maxf = feret.max(label_img_copy, edge=True)
            minf = feret.min(label_img_copy, edge=True)

            writer.writerow([cell_count, region.area_filled, maxf, minf])
            cell_count += 1

        

def main():

    if len(sys.argv) == 1:
        sys.exit(f"You must provide a list of area filtered images")
    
    os.makedirs('labeled_images', exist_ok=True)
    os.makedirs('labeled_csv', exist_ok=True)

    for area_filtered_filename in sys.argv[1:]:
        original_filename = area_filtered_filename.replace('-area_filtered', '')

        predicted_mask = cv2.imread(area_filtered_filename)
        predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_BGR2GRAY)

        original_image = cv2.imread(original_filename)

        label_image(predicted_mask, original_image, original_filename)


if __name__ == "__main__":
    main()




