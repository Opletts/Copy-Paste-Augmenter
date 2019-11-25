import os
import cv2
import numpy as np
import dataset_reader as dr

from labels import *

img_path = "Path to images"
lbl_path = "Path to labels"

data = dr.DatasetReader(image_path=img_path, label_path=lbl_path)

object_name = "person"
class_id = names2labels[object_name].color
os.makedirs(object_name, exist_ok=True)

## You can set the padding as well as minimum height and width for the object
## to be saved
padding = 0
min_h, min_w = 50, 50

count = 0
for i in range(len(data)):
    image, label = data[i]
    height, width, _ = image.shape

    mask = cv2.inRange(label, class_id, class_id)
    image = cv2.bitwise_and(image, image, mask=mask)
    contours, _ = cv2.findContours(mask, 1, 2)

    instances = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        instances.append([x, y, x+w, y+h])

    deletions = []
    for x1, y1, x2, y2 in instances:
        for a, b, c, d in instances:
            if x1 > a and y1 > b and x2 < c and y2 < d:
                deletions.append([x1, y1, x2, y2])
                break

    instances = [x for x in instances if x not in deletions]

    for x1, y1, x2, y2 in instances:
        roi = image[y1-padding:y2+padding, x1-padding:x2+padding]
        roi_label = mask[y1-padding:y2+padding, x1-padding:x2+padding]

        roi_h, roi_w, _ = roi.shape
        expected_h = y2 - y1 + 2 * padding
        expected_w = x2 - x1 + 2 * padding

        diff_h, diff_w = 0, 0
        if roi_h != expected_h or roi_w != expected_w:
            diff_h = expected_h - roi_h
            diff_w = expected_w - roi_w

        temp = np.zeros((expected_h - diff_h, expected_w - diff_w), np.uint8)
        temp[np.where(temp == 0)] = 255

        temp = cv2.bitwise_xor(temp, roi_label)
        roi = cv2.bitwise_and(roi, roi, mask=temp)

        if x2 - x1 > min_w and y2 - y1 > min_h:
            class_ix = image[y1:y2, x1:x2]
            count += 1

            # Saved as person/person{1,2,3..}.png
            cv2.imwrite(object_name + '/' + object_name + str(count) + '.png', class_ix)

            # Uncomment to display the classes that are being saved.
            # cv2.imshow("test", class_ix)
            # cv2.waitKey(0)
