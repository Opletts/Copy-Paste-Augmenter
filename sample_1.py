import cv2
import dataset_reader as dr

from labels import *
from Augmenter import utils
from Augmenter import base_augmenter as ba

img_path = "Path to images"
lbl_path = "Path to labels"

aug_class_path = "Path to object images extracted by class_extractor.py"

data = dr.DatasetReader(image_path=img_path, label_path=lbl_path)

# Modify the class_id and placement_id for other classes, refer labels.py
class_id = names2labels["person"].color
placement_id = (names2labels["sidewalk"].color, names2labels["terrain"].color,
                names2labels["parking"].color, names2labels["road"].color,
                names2labels["ground"].color)

# Modify horizon_line and max_height as per your requirement
rows, cols, _ = data[0][0].shape
horizon_line = int(rows * 0.4)
max_height = int(rows * 0.8)

for i in range(len(data)):
    image, label = data[i]

    aug = ba.BaseAugmenter(image, label, class_id, placement_id=placement_id,
                           horizon_line=horizon_line, max_height=max_height)

    # aug.set_limit((0.6, 0.8))
    img, lbl = aug.place_class(2, aug_class_path)

    cv2.imshow("image", cv2.resize(img, (1024, 512)))
    cv2.imshow("label", cv2.resize(lbl, (1024, 512)))
    cv2.imshow("placement", cv2.resize(utils.viz_placement(aug), (1024, 512)))
    cv2.imshow("scaling_triangle", cv2.resize(utils.viz_scaling_triangle(aug), (1024, 512)))
    cv2.waitKey(0)
