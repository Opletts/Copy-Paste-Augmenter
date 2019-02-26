import os
import cv2
import random
import numpy as np
import imgaug as ia
from Augmenter import utils

class BaseAugmenter(object):
    """Parent class for all object types in the image that can be augmented"""

    def __init__(self, image, label, class_id, placement_id):
        self.image = image.copy()
        self.label = label.copy()

        self.image_copy = image.copy()
        self.label_copy = label.copy()

        self.class_id = class_id
        self.fake_class_id = [i+1 for i in list(class_id)]

        self.row_value, self.col_value = utils.threshold(image, label, placement_id)
        self.col_value = self.col_value[self.row_value - self.max_height > 0]
        self.row_value = self.row_value[self.row_value - self.max_height > 0]

        self.triangle_init()
        ## For random placement of signs
        # self.row_value = range(self.rows)
        # self.col_value = range(self.cols)

        self.class_placement = []
        self.get_class_pos(self.class_id)

    def reset(self):
        self.image = self.image_copy.copy()
        self.label = self.label_copy.copy()

    def scale(self, x, y, sigma):
        x_intersect = (y - self.c_intercept) / self.slope
        cur_triangle_side = np.sqrt(np.power(self.max_height - y, 2) + np.power(self.cols/2 - x_intersect, 2))
        ratio = cur_triangle_side / (self.main_traingle_side + sigma)

        ## Random scaling of signs
        # ratio = random.random()
        return ratio

    def viz_scaling_triangle(self, img):
        triangle = np.array([[0, self.rows], [self.cols/2, self.max_height], [self.cols, self.rows]], np.int32)
        temp = img.copy()
        cv2.fillConvexPoly(temp, triangle, (255, 255, 0))
        cv2.addWeighted(temp, 0.3, img, 0.7, 0, temp)

        return temp

    def create_roi(self, x, y, class_img, y_displacement=0, extra_class_id=0):
        height, width, _ = class_img.shape
        roi_x_start = x - width // 2
        roi_x_end = x + 1 + width // 2

        roi = self.image[y-height:y, roi_x_start:roi_x_end]
        roi_label = self.label[y-height:y, roi_x_start:roi_x_end]

        ## Padding around the roi for blurring the edges of the class image properly
        padding = 10
        pad_roi = self.image[y-height-padding:y+padding, roi_x_start-padding:roi_x_end+padding]
        pad_class_img = np.uint8(np.zeros((height+2*padding, width+2*padding, 3)))
        pad_class_img[padding:padding+height, padding:padding+width] = class_img

        bb_curr = ia.BoundingBox(x1=roi_x_start, y1=y-height, x2=roi_x_end, y2=y)

        if roi.shape == (height, width, 3) and pad_class_img.shape == pad_roi.shape:
            for i in self.class_placement:
                bb_i = ia.BoundingBox(x1=i[0], y1=i[1], x2=i[2], y2=i[3])
                iou = bb_curr.iou(bb_i)
                if iou:
                    return 1

            self.class_placement.append([roi_x_start, y-height, roi_x_end, y, 0])
            if extra_class_id == 0:
                roi_label[np.where(class_img[:, :, 0] > 10)] = self.fake_class_id
            else:
                roi_label[np.where(class_img[:, :, 0] > 10)] = extra_class_id

            hist_template = self.image[y-height:y+y_displacement, roi_x_start:roi_x_end]
            roi = utils.blend(roi, class_img, hist_template, 1)

            self.image[y-height:y, roi_x_start:roi_x_end] = roi

            utils.smooth_edges(pad_roi, pad_class_img)
            return 0

        else:
            return 1

    def get_class_pos(self, class_id):
        mask = cv2.inRange(self.label, class_id, class_id)
        _, contours, _ = cv2.findContours(mask, 1, 2)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            self.class_placement.append([x, y, x+w, y+h, 1])

    def place_class(self, num_class, path):
        while num_class != 0 and len(self.row_value):
            all_class_imgs = os.listdir(path)
            class_img = cv2.imread(os.path.join(path, random.choice(all_class_imgs)))

            class_height, class_width, _ = class_img.shape

            index = random.randint(0, len(self.row_value) - 1)
            x, y = self.col_value[index], self.row_value[index]

            self.row_value = np.delete(self.row_value, index)
            self.col_value = np.delete(self.col_value, index)

            ## Calculate ratio and scale the class image
            sigma = 20
            ratio = self.scale(x, y, sigma)

            init_scale = float(self.max_height) / class_height

            scaled_class_width = int(class_width * init_scale * ratio)
            scaled_class_height = int(self.max_height * ratio)

            ## Should be atleast 20 px tall
            min_px = 20
            if scaled_class_height < min_px:
                continue

            ## Width needs to be odd for equal splitting about mid point
            scaled_class_width -= 1 if scaled_class_width % 2 == 0 else 0

            scaled_class_img = cv2.resize(class_img, (scaled_class_width, scaled_class_height), interpolation=cv2.INTER_CUBIC)

            class_err_code = self.place_extra_class(x, y, scaled_class_img)

            if class_err_code == 1:
                continue

            num_class -= 1

        return self.image, self.label

    def place_extra_class(self, x, y, scaled_class_img):
        """Function to be overloaded"""

        class_err_code = self.create_roi(x, y, scaled_class_img)
        if class_err_code:
            return 1

        return 0
