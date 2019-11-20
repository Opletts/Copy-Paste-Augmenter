import cv2
import dataset_reader as DR

from labels import *
from Augmenter import utils
from Augmenter import base_augmenter as BA

class PeopleAugmenter(BA.BaseAugmenter):
    """Test class for augmentation"""

    def __init__(self, image, label, class_id, placement_id=None):
        self.rows, self.cols, _ = image.shape
        self.horizon_line = int(self.rows * 0.4)

        self.max_height = int(self.rows * 0.8)

        super(PeopleAugmenter, self).__init__(image, label, class_id, placement_id)

img_path = "Path to images"
lbl_path = "Path to labels"

aug_class_path = "Path to object images to be augmented in"

data = DR.DatasetReader(image_path=img_path, label_path=lbl_path)

class_id = names2labels["person"].color
placement_id = names2labels["sidewalk"].color

for i in range(len(data)):
    image, label = data[i]
    
    aug = PeopleAugmenter(image, label, class_id, placement_id)
    img, lbl = aug.place_class(1, aug_class_path)

    cv2.imshow("image", img)
    cv2.imshow("label", lbl)
    cv2.waitKey(0)
