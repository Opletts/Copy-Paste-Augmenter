import os
import cv2

import dataset_reader as DR

img_path = "/home/opletts/Stuff/Cityscapes/leftImg8bit/val"
lbl_path = "/home/opletts/Stuff/Cityscapes/gtFine/val"

data = DR.DatasetReader(image_path=img_path, label_path=lbl_path)

os.mkdir('Signs')
count = 0

def split_data(lower, upper, class_value):
    global count
    for i in xrange(lower, upper):
        image, label = data[i]

        mask = cv2.inRange(label, class_value, class_value)
        image = cv2.bitwise_and(image, image, mask=mask)

        _, contours, _ = cv2.findContours(mask, 1, 2)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 50:
                class_ix = image[y:y+h, x:x+w]
                count += 1
                cv2.imwrite('./Signs/Sign' + str(count) + '.png', class_ix)

                ## Uncomment to display signs that are being saved.
                # cv2.imshow("test", class_ix)
                # cv2.waitKey(0)

split_data(0, len(data), (0, 220, 220))

print count + " Signs"
