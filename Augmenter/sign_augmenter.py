import os
import cv2
import random
import numpy as np
import imgaug as ia

from Augmenter import utils

class SignAugmenter:
    """A class for artificially placing signs into images for augmentation"""

    def __init__(self, image, label):
        self.image = image.copy()
        self.label = label.copy()

        self.image_copy = image.copy()
        self.label_copy = label.copy()

        self.rows, self.cols, _ = image.shape
        self.max_height = int(self.rows * 0.4)

        self.row_value, self.col_value = utils.threshold(self.image, self.label, ((152, 251, 152), (232, 35, 244)))
        self.col_value = self.col_value[self.row_value - self.max_height > 0]
        self.row_value = self.row_value[self.row_value - self.max_height > 0]

        self.main_traingle_side = np.sqrt(np.power(self.max_height - self.rows, 2) + np.power(self.cols/2, 2))
        self.slope = float(self.max_height - self.rows) / (self.cols / 2)
        self.c_intercept = self.rows

        self.sign_placement = []
        self.get_sign_pos()


    def reset(self):
        self.image = self.image_copy.copy()
        self.label = self.label_copy.copy()

    def scale(self, x, y, sigma):
        x_intersect = (y - self.c_intercept) / self.slope
        cur_triangle_side = np.sqrt(np.power(self.max_height - y, 2) + np.power(self.cols/2 - x_intersect, 2))
        ratio = cur_triangle_side / (self.main_traingle_side + sigma)

        return ratio

    def viz_scaling_triangle(self, img):
        triangle = np.array([[0, self.rows], [self.cols/2, self.max_height], [self.cols, self.rows]], np.int32)
        temp = img.copy()
        cv2.fillConvexPoly(temp, triangle, (255, 255, 0))
        cv2.addWeighted(temp, 0.3, img, 0.7, 0, temp)

        return temp

    def create_roi(self, x, y, class_img, flag=1):
        height, width, _ = class_img.shape
        roi_x_start = x - width // 2
        roi_x_end = x + 1 + width // 2

        roi = self.image[y-height:y, roi_x_start:roi_x_end]

        pad_roi = self.image[y-height-10:y+10, roi_x_start-10:roi_x_end+10]
        pad_class_img = np.uint8(np.zeros((height+20, width+20, 3)))
        pad_class_img[10:10+height, 10:10+width] = class_img

        bb_curr = ia.BoundingBox(x1=roi_x_start, y1=y-height, x2=roi_x_end, y2=y)

        if roi.shape == (height, width, 3) and pad_class_img.shape == pad_roi.shape:
            hist_template = 0
            if flag:
                for i in self.sign_placement:
                    bb_i = ia.BoundingBox(x1=i[0], y1=i[1], x2=i[2], y2=i[3])
                    iou = bb_curr.iou(bb_i)
                    if iou > 0.05:
                        return 1

                self.sign_placement.append([roi_x_start, y-height, roi_x_end, y, 0])
                hist_template = self.image[y-height:y+int(1.5*height), roi_x_start:roi_x_end]

            roi = utils.blend(roi, class_img, hist_template, flag)
            self.image[y-height:y, roi_x_start:roi_x_end] = roi

            utils.smooth_edges(pad_roi, pad_class_img)

            return 0
        else:
            return 1

    def get_sign_pos(self):
        mask = cv2.inRange(self.label, (0, 220, 220), (0, 220, 220))
        _, contours, _ = cv2.findContours(mask, 1, 2)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            self.sign_placement.append([x, y, x+w, y+h, 1])

    def place_sign(self, num_signs):
        self.reset()
        while num_signs != 0 and len(self.row_value):
            all_signs = os.listdir('./Signs/Usable/')
            sign = cv2.imread(os.path.join('./Signs/Usable', random.choice(all_signs)))
            # all_signs = os.listdir('./Signs/Signs/')
            # sign = cv2.imread(os.path.join('./Signs/Signs', random.choice(all_signs)))
            sign_height, sign_width, _ = sign.shape

            index = random.randint(0, len(self.row_value) - 1)
            x, y = self.col_value[index], self.row_value[index]

            self.row_value = np.delete(self.row_value, index)
            self.col_value = np.delete(self.col_value, index)

            ratio = self.scale(x, y, 20)

            # Need to resize sign to max_limits first then apply scaling
            init_scale = float(self.max_height) / sign_height

            scaled_sign_width = int(sign_width * init_scale * ratio)
            scaled_sign_height = int(self.max_height * ratio)
            if scaled_sign_height < 20:
                continue

            scaled_sign_width -= 1 if scaled_sign_width % 2 == 0 else 0

            scaled_sign = cv2.resize(sign, (scaled_sign_width, scaled_sign_height), interpolation=cv2.INTER_CUBIC)

            all_poles = os.listdir('./Poles')
            pole = cv2.imread(os.path.join('./Poles', random.choice(all_poles)))

            pole_height = int(1.5 * scaled_sign_height)
            pole_width = int(0.1 * scaled_sign_width)
            pole_width -= 1 if pole_width % 2 == 0 else 0
            pole_width = 1 if pole_width < 0 else pole_width

            scaled_pole = cv2.resize(pole, (pole_width, pole_height), interpolation=cv2.INTER_CUBIC)

            sign_err_code = self.create_roi(x, y-pole_height, scaled_sign)
            if sign_err_code:
                continue
            # cv2.rectangle(self.image, (x-scaled_sign_width//2, y-pole_height-scaled_sign_height), (x+scaled_sign_width/2, y-pole_height), (255, 255, 0), 2)
            pole_err_code = self.create_roi(x, y, scaled_pole, 0)

            num_signs -= 1

        return self.image

    # def add_sign(self):
    #
    #     return image
