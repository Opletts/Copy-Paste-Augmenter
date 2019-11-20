import os
import cv2
import random
import numpy as np
from Augmenter import utils

class BaseAugmenter(object):
    """Parent class for all object types in the image that can be augmented"""

    def __init__(self, image, label, class_id, placement_id=None):
        self.called = 0
        self.limits = None
        self.counter = 0

        self.image = image.copy()
        self.label = label.copy()

        self.image_copy = image.copy()
        self.label_copy = label.copy()

        self.class_id = class_id
        self.fake_class_id = [i if i == 255 else i+1 for i in class_id]

        ## For random placement of object class
        self.placement_id = placement_id

        if placement_id != None:
            self.row_value, self.col_value = utils.threshold(image, label, placement_id)
            self.col_value = self.col_value[self.row_value - self.horizon_line > 0]
            self.row_value = self.row_value[self.row_value - self.horizon_line > 0]

            self.triangle_init()

        else:
            self.row_value = range(self.rows)
            self.col_value = range(self.cols)

        self.copy_row_value = self.row_value
        self.copy_col_value = self.col_value

        self.class_placement = utils.get_class_pos(self.label, self.class_id)

    def triangle_init(self):
        self.main_traingle_side = np.sqrt(np.power(self.horizon_line - self.rows, 2) + np.power(self.cols/2, 2))
        self.slope = float(self.horizon_line - self.rows) / (self.cols / 2)
        self.c_intercept = self.rows

    def reset(self):
        self.image = self.image_copy.copy()
        self.label = self.label_copy.copy()

    def set_limit(self, limit, color=None):
        """Function to filter the placement array to constrain the number of
        augmented pixels per image.
        range = (lower_percent, higher_percent)
                 percentage of the total image height requested
        """
        self.limits = limit

        sigma = 0
        self.col_value = self.copy_col_value
        self.row_value = self.copy_row_value

        min_scaled_class_height, max_scaled_class_height = np.array(limit) * self.rows

        min_ratio = float(min_scaled_class_height) / self.max_height
        max_ratio = float(max_scaled_class_height) / self.max_height

        min_cur_triangle_side = min_ratio * (self.main_traingle_side + sigma)
        max_cur_triangle_side = max_ratio * (self.main_traingle_side + sigma)

        y_min = (min_cur_triangle_side * (self.rows - self.horizon_line) /
             self.main_traingle_side + self.horizon_line)

        y_max = (max_cur_triangle_side * (self.rows - self.horizon_line) /
             self.main_traingle_side + self.horizon_line)

        self.col_value = self.col_value[np.logical_and(self.row_value > y_min, self.row_value < y_max)]
        self.row_value = self.row_value[np.logical_and(self.row_value > y_min, self.row_value < y_max)]

    def scale(self, x, y, class_img):
        ## Regularizing term
        sigma = 0

        ## Random scaling of object class if None
        if self.placement_id != None:
            x_intersect = (y - self.c_intercept) / self.slope
            cur_triangle_side = np.sqrt(np.power(self.horizon_line - y, 2) + np.power(self.cols / 2 - x_intersect, 2))
            ratio = cur_triangle_side / (self.main_traingle_side + sigma)

        else:
            ratio = random.random()

        class_height, class_width, _ = class_img.shape

        init_scale = float(self.max_height) / class_height

        scaled_class_width = int(class_width * init_scale * ratio)
        scaled_class_height = int(self.max_height * ratio)

        return scaled_class_width, scaled_class_height

    ## flag enables / disables poisson blending, default ON
    def create_roi(self, x, y, class_img, y_displacement=0, extra_class_id=0, flag=1):
        height, width, _ = class_img.shape
        roi_x_start = x - width // 2
        roi_x_end = x + 1 + width // 2

        x1, y1, x2, y2 = roi_x_start, y-height, roi_x_end, y

        roi = self.image[y1:y2, x1:x2]
        roi_label = self.label[y1:y2, x1:x2]

        ## Padding around the roi for blurring the edges of the class image properly
        padding = 10

        pad_roi = self.image[y1-padding:y2+padding, x1-padding:x2+padding]
        pad_class_img = np.uint8(np.zeros((height+2*padding, width+2*padding, 3)))
        pad_class_img[padding:padding+height, padding:padding+width] = class_img

        if roi.shape == (height, width, 3) and pad_class_img.shape == pad_roi.shape:
            for a1, b1, a2, b2, _ in self.class_placement:
                iou = utils.get_iou([x1, y1, x2, y2], [a1, b1, a2, b2])
                if iou > 0.4:
                    return 1

            self.class_placement.append([x1, y1, x2, y2, 0])
            if extra_class_id == 0:
                roi_label[np.where(class_img[:, :, 0] != 0)] = self.fake_class_id
            else:
                roi_label[np.where(class_img[:, :, 0] > 10)] = extra_class_id

            hist_template = self.image[y1:y2+y_displacement, x1:x2]
            roi = utils.blend(pad_roi, pad_class_img, hist_template, flag)

            self.image[y1-padding:y2+padding, x1-padding:x2+padding] = roi
            utils.smooth_edges(pad_roi, pad_class_img)

            return 0

        else:
            return 1

    def place_class(self, num_class, path):
        self.called += 1

        updated_img = self.image.copy()
        updated_lbl = self.label.copy()

        while num_class != 0 and len(self.row_value):
            all_class_imgs = os.listdir(path)
            class_img = cv2.imread(os.path.join(path, random.choice(all_class_imgs)))

            class_height, class_width, _ = class_img.shape

            index = random.randint(0, len(self.row_value) - 1)
            x, y = self.col_value[index], self.row_value[index]

            self.row_value = np.delete(self.row_value, index)
            self.col_value = np.delete(self.col_value, index)

            ## Calculate ratio and scale the class image
            scaled_class_width, scaled_class_height = self.scale(x, y, class_img)

            ## Should be atleast 20 px tall
            min_px = 10
            if scaled_class_height < min_px:
                continue

            ## Width needs to be odd for equal splitting about mid point
            scaled_class_width -= 1 if scaled_class_width % 2 == 0 else 0

            scaled_class_img = cv2.resize(class_img, (scaled_class_width, scaled_class_height),
                                            interpolation=cv2.INTER_CUBIC)

            class_err_code = self.place_extra_class(x, y, scaled_class_img)

            if class_err_code == 1:
                self.image = updated_img.copy()
                self.label = updated_lbl.copy()
                continue

            updated_img = self.image.copy()
            updated_lbl = self.label.copy()
            num_class -= 1
            self.counter = 1

        if self.limits != None and len(self.copy_row_value) and num_class != 0:
            diff = self.limits[1] - self.limits[0]
            lower_limit = round(self.limits[0] - diff, 1)
            upper_limit = self.limits[0]

            if lower_limit < 0:
                lower_limit = 0.0

            if upper_limit != 0:
                self.set_limit((lower_limit, upper_limit))
                self.place_class(num_class, path)

        return self.image, self.label

    def place_extra_class(self, x, y, scaled_class_img):
        """Function to be overloaded"""

        class_err_code = self.create_roi(x, y, scaled_class_img)
        if class_err_code:
            return 1

        return 0
