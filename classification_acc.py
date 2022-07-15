import sys
import cv2
from skimage.measure import label, regionprops

def classification_acc(truth_mask, predicted_mask):

    true_pos = 0
    false_pos = 0
    false_neg = 0
    true_total = 0
    label_truth = label(truth_mask)
    label_predicted = label(predicted_mask)

    for region in regionprops(label_predicted):
        x, y = region.centroid
        x = int(x)
        y = int(y)
        if truth_mask[x, y] == 255:
            true_pos += 1
        else:
            false_pos += 1

    for region in regionprops(label_truth):
        x, y = region.centroid
        x = int(x)
        y = int(y)
        true_total += 1
        if predicted_mask[x, y] != 255:
            false_neg += 1
    
    print(f"True Positive Rate: {true_pos}/{true_pos+false_neg} = {true_pos/(true_pos+false_neg)}")
    print(f"False Positives: {false_pos}")
    print(f"False Negative Rate: {false_neg}/{true_pos+false_neg} = {false_neg/(true_pos+false_neg)}")

    



def main():

    if len(sys.argv) != 3:
        sys.exit(f"Usage: {sys.argv[0]} <truth mask> <predicted mask>")
    
    truth_mask = cv2.imread(sys.argv[1])
    truth_mask = cv2.cvtColor(truth_mask, cv2.COLOR_BGR2GRAY)
    predicted_mask = cv2.imread(sys.argv[2])
    predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_BGR2GRAY)

    classification_acc(truth_mask, predicted_mask)


if __name__ == "__main__":
    main()
