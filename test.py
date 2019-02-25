import os
import cv2
import random
import numpy as np
import imgaug as ia
import dataset_reader as DR

from Augmenter import utils
from Augmenter import base_augmenter as BA

class SignAugmenter(BA.BaseAugmenter):
    """Test class for augmentation"""

    def __init__(self, image, label, class_id, placement_id):
        self.rows, self.cols, _ = image.shape
        self.max_height = int(self.rows * 0.4)

        super(SignAugmenter, self).__init__(image, label, class_id, placement_id)

    def place_extra_class(self, x, y, scaled_class_img):
        scaled_class_height, scaled_class_width, _ = scaled_class_img.shape
        all_poles = os.listdir('./Poles')
        pole = cv2.imread(os.path.join('./Poles', random.choice(all_poles)))

        pole_height = int(1.5 * scaled_class_height)
        pole_width = int(0.1 * scaled_class_width)
        pole_width -= 1 if pole_width % 2 == 0 else 0
        pole_width = 1 if pole_width <= 0 else pole_width

        scaled_pole = cv2.resize(pole, (pole_width, pole_height), interpolation=cv2.INTER_CUBIC)

        class_err_code = self.create_roi(x, y-pole_height, scaled_class_img, pole_height)
        if class_err_code:
            return 1

        pole_err_code = self.create_roi(x, y, scaled_pole, 0, [143, 143, 143])

        return 0

placement_id = ((152, 251, 152), (232, 35, 244))
# placement_id = (128, 64, 128)
class_id = (0, 220, 220)
img_path = "/home/opletts/Stuff/Cityscapes/leftImg8bit/train"
lbl_path = "/home/opletts/Stuff/Cityscapes/gtFine/train"

data = DR.DatasetReader(image_path=img_path, label_path=lbl_path)

for i in range(len(data)):
    image, label = data[i]
    image = cv2.resize(image, (1024, 512))
    label = cv2.resize(label, (1024, 512))
    aug = SignAugmenter(image, label, class_id, placement_id)
    for j in range(1, 3):
        img, lbl = aug.place_class(1, './Signs/Usable')
        cv2.imshow("image", img)
        cv2.imshow("label", lbl)
        cv2.waitKey(0)

# image = cv2.imread("/home/opletts/Data/img.png")
# label = cv2.imread("/home/opletts/Data/lbl.png")
#
# aug = SignAugmenter(image, label, class_id, placement_id)
# while 1:
#     img, lbl = aug.place_class(1, './Signs/Usable')
#     cv2.imshow("Image", cv2.resize(img, (1024, 512)))
#     cv2.imshow("Label", cv2.resize(lbl, (1024, 512)))
#     cv2.waitKey(0)
