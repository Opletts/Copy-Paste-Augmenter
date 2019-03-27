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

    def __init__(self, image, label, class_id, placement_id=None):
        self.rows, self.cols, _ = image.shape
        self.horizon_line = int(self.rows * 0.4)

        self.max_height = int(self.rows * 0.4)

        super(SignAugmenter, self).__init__(image, label, class_id, placement_id)

    def place_extra_class(self, x, y, scaled_class_img):
        scaled_class_height, scaled_class_width, _ = scaled_class_img.shape
        all_poles = os.listdir('./Poles')
        pole = cv2.imread(os.path.join('./Poles', random.choice(all_poles)))

        ## SAMPLE
        ## If scaling for this different and needs to be calculated
        #
        # self.max_height = int(self.rows * 0.6)
        # scaled_class_width, scaled_class_height = self.scale(x, y, class_img)
        # min_px = 20
        # if scaled_class_height < min_px:
        #     continue
        # scaled_class_width -= 1 if scaled_class_width % 2 == 0 else 0
        # scaled_class_img = cv2.resize(class_img, (scaled_class_width, scaled_class_height), interpolation=cv2.INTER_CUBIC)

        ## RESET scaling values for og class if scaling is changed
        # self.max_height = int(self.rows * 0.4)

        init_height = random.uniform(1.3, 1.7)
        init_width = random.uniform(0.08, 0.13)

        pole_height = int(init_height * scaled_class_height)
        pole_width = int(init_width * scaled_class_width)
        pole_width -= 1 if pole_width % 2 == 0 else 0
        pole_width = 1 if pole_width <= 0 else pole_width

        scaled_pole = cv2.resize(pole, (pole_width, pole_height), interpolation=cv2.INTER_CUBIC)

        sign_err_code = self.create_roi(x, y-pole_height, scaled_class_img, pole_height)
        if class_err_code:
            return 1

        pole_err_code = self.create_roi(x, y, scaled_pole, 0, [154, 154, 154], 1)
        if pole_err_code:
            return 1

        return 0

placement_id = ((152, 251, 152), (232, 35, 244)) # pavements
# placement_id = (128, 64, 128) # road
class_id = (0, 220, 220) # signs
# class_id = (60, 20, 220)    # people
img_path = "/home/opletts/Stuff/Cityscapes/leftImg8bit/val"
lbl_path = "/home/opletts/Stuff/Cityscapes/gtFine/val"

save_path = "/media/opletts/New Volume/AugmentedData/IMG500/Set3/Signs/"

data = DR.DatasetReader(image_path=img_path, label_path=lbl_path)

for i in range(len(data)):
    image, label = data[i]
    aug = SignAugmenter(image, label, class_id, placement_id)
    for j in range(1, 5):
        print(i, " - ", j)
        img, lbl = aug.place_class(1, './Signs/Usable')
        # img, lbl = aug.place_class(1, './People/')

        save_img = save_path  + str(j) + "_Aug/image/"
        save_lbl = save_path + str(j) + "_Aug/label/"

        # save_img = "/media/opletts/New Volume/AugmentedData/IMG500/random_" + str(j) + "_Aug/image/"
        # save_lbl = "/media/opletts/New Volume/AugmentedData/IMG500/random_" + str(j) + "_Aug/label/"

        # cv2.imshow("trian", cv2.resize(aug.viz_scaling_triangle(img), (1024, 512)))
        cv2.imshow("image", cv2.resize(img, (1024, 512)))
        cv2.imshow("label", cv2.resize(lbl, (1024, 512)))
        cv2.waitKey(0)

        # cv2.imwrite(save_img + "image" + str(i+1) + ".png", img)
        # cv2.imwrite(save_lbl + "label" + str(i+1) + ".png", lbl)
