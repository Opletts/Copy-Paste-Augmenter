import os
import cv2
import random
import numpy as np
from Augmenter import utils


class BaseAugmenter(object):
    """
    Parent class for all object types in the image that can be augmented
    """

    def __init__(self, image, label, class_id, placement_id=None, horizon_line=None,
                 max_height=None, max_iou=0.4, padding=10, min_px=10, sigma=0):
        """
        Constructor

        image: image to be augmented
        label: semantic label to be modified
        class_id: BGR value of object to be copied into the image
        placement_id: possible locations for the object to be placed
        horizon_line: location of the horizon for scaling accurately
        max_height: size of the object if it were copied in an area closest to the camera
        max_iou: maximum overlap allowed between objects of same class
        padding: padding applied around roi for optimal blurring
        min_px: number of pixels tall the scaled object should be to consider it a valid copy paste
        sigma: increase/decrease the value to decrease/increase the scaling ratio
        """

        self.called = 0
        self.counter = 0
        self.limits = None

        self.sigma = sigma
        self.max_iou = max_iou
        self.padding = padding
        self.min_px = min_px

        self.rows, self.cols, _ = image.shape

        self.image = image.copy()
        self.label = label.copy()

        self.class_id = class_id
        self.fake_class_id = [i if i == 255 else i + 1 for i in class_id]

        self.placement_id = placement_id
        self.horizon_line = horizon_line
        self.max_height = max_height

        if self.max_height is None:
            self.max_height = self.rows * 0.8

        if placement_id is not None:
            self.row_value, self.col_value = utils.threshold(image, label, placement_id)

        else:
            self.row_value, self.col_value = np.mgrid[0:len(range(self.rows)), 0:len(range(self.cols))]
            self.row_value, self.col_value = self.row_value.ravel(), self.col_value()

        if self.horizon_line is not None:
            self.col_value = self.col_value[self.row_value - self.horizon_line > 0]
            self.row_value = self.row_value[self.row_value - self.horizon_line > 0]

            # Initialize scaling triangle
            #           pt1
            #           .
            #     pt2 .   . pt3
            # pt1 = main_triangle_side = (horizon_line, cols / 2)
            # pt2 = (rows, 0)

            self.main_triangle_side = np.sqrt(np.power(self.horizon_line - self.rows, 2) + np.power(self.cols / 2, 2))
            self.slope = float(self.horizon_line - self.rows) / (self.cols / 2)
            self.y_intercept = self.rows

        self.copy_row_value = self.row_value
        self.copy_col_value = self.col_value

        self.class_placement = utils.get_class_pos(self.label, self.class_id)

    def set_limit(self, limit):
        """
        Filters the placement array to constrain the number of
        augmented pixels per image.

        limit = (lower_percent, higher_percent)
                 percentage of the total image height requested
        """
        assert self.horizon_line is not None, "Can't call set_limit without setting a horizon line!"

        self.limits = limit

        self.col_value = self.copy_col_value
        self.row_value = self.copy_row_value

        min_scaled_class_height, max_scaled_class_height = np.array(limit) * self.rows

        min_ratio = float(min_scaled_class_height) / self.max_height
        max_ratio = float(max_scaled_class_height) / self.max_height

        min_cur_triangle_side = min_ratio * (self.main_triangle_side + self.sigma)
        max_cur_triangle_side = max_ratio * (self.main_triangle_side + self.sigma)

        y_min = (min_cur_triangle_side * (self.rows - self.horizon_line) /
                 self.main_triangle_side + self.horizon_line)

        y_max = (max_cur_triangle_side * (self.rows - self.horizon_line) /
                 self.main_triangle_side + self.horizon_line)

        self.col_value = self.col_value[np.logical_and(self.row_value > y_min, self.row_value < y_max)]
        self.row_value = self.row_value[np.logical_and(self.row_value > y_min, self.row_value < y_max)]

    def scale(self, x, y, class_img):
        """
        Scales the object according to user inputs for copying

        x: x co-ord of selected point to copy to (col)
        y: y co-ord of selected point to copy to (row)
        class_img: object image to copy
        """

        # Modify sigma if you want to further reduce or increase the ratio
        # Random scaling of object class if placement_id is None

        if self.horizon_line is not None:
            x_intersect = (y - self.y_intercept) / self.slope
            cur_triangle_side = np.sqrt(np.power(self.horizon_line - y, 2) + np.power(self.cols / 2 - x_intersect, 2))
            ratio = cur_triangle_side / (self.main_triangle_side + self.sigma)

        else:
            ratio = random.random()

        class_height, class_width, _ = class_img.shape

        init_scale = float(self.max_height) / class_height

        scaled_class_width = int(class_width * init_scale * ratio)
        scaled_class_height = int(self.max_height * ratio)

        return scaled_class_width, scaled_class_height

    def create_roi(self, x, y, class_img, extra_class_id=0, flag=1):
        """
        Creates the required roi for the object and copies it into the image

        x: x co-ord of selected point to copy to (col)
        y: y co-ord of selected point to copy to (row)
        class_img: object image to copy
        extra_class_id: BGR value if copying multiple objects in
        flag: enables poisson blending
        """

        height, width, _ = class_img.shape
        roi_x_start = x - width // 2
        roi_x_end = x + 1 + width // 2

        x1, y1, x2, y2 = roi_x_start, y - height, roi_x_end, y

        roi = self.image[y1:y2, x1:x2]
        roi_label = self.label[y1:y2, x1:x2]

        # Padding around the roi for blurring the edges of the class image properly
        pad = self.padding

        pad_roi = self.image[y1 - pad:y2 + pad, x1 - pad:x2 + pad]
        pad_class_img = np.uint8(np.zeros((height + 2 * pad, width + 2 * pad, 3)))
        pad_class_img[pad:pad + height, pad:pad + width] = class_img

        if roi.shape == (height, width, 3) and pad_class_img.shape == pad_roi.shape:
            for a1, b1, a2, b2, _ in self.class_placement:
                iou = utils.get_iou([x1, y1, x2, y2], [a1, b1, a2, b2])

                # Control the max amount of overlap allowed
                if iou > self.max_iou:
                    return 1

            self.class_placement.append([x1, y1, x2, y2, 0])
            if extra_class_id == 0:
                roi_label[np.where(class_img[:, :, 0] != 0)] = self.fake_class_id
            else:
                roi_label[np.where(class_img[:, :, 0] != 0)] = extra_class_id

            roi = utils.blend(pad_roi, pad_class_img, flag)
            self.image[y1 - pad:y2 + pad, x1 - pad:x2 + pad] = roi
            utils.smooth_edges(pad_roi, pad_class_img)

            return 0

        else:
            return 1

    def place_class(self, num_class, path):
        """
        Copy the required amount of objects into the image

        num_class: number of objects to be copied per image
        path: path to the folder containing object images extracted
        """

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

            scaled_class_width, scaled_class_height = self.scale(x, y, class_img)

            # Should be at least min_px tall, change accordingly
            if scaled_class_height < self.min_px:
                continue

            # Width needs to be odd for equal splitting about mid point
            scaled_class_width -= 1 if scaled_class_width % 2 == 0 else 0

            scaled_class_img = cv2.resize(class_img, (scaled_class_width, scaled_class_height),
                                          interpolation=cv2.INTER_CUBIC)

            class_err_code = self.place_extra_class(x, y, scaled_class_img)

            if class_err_code == 1:
                self.image = updated_img.copy()
                self.label = updated_lbl.copy()
                continue

            else:
                updated_img = self.image.copy()
                updated_lbl = self.label.copy()
                num_class -= 1
                self.counter = 1

        if self.limits is not None and len(self.copy_row_value) and num_class != 0:
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
        """
        Function to be overloaded
        """

        class_err_code = self.create_roi(x, y, scaled_class_img)
        if class_err_code:
            return 1

        return 0
