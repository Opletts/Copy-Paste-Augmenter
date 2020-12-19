import cv2
import numpy as np


def threshold(image, label, value):
    if type(value[0]) == int:
        mask = cv2.inRange(label, value, value)
    else:
        mask = 0
        for i in value:
            mask += cv2.inRange(label, i, i)

    placement = cv2.bitwise_and(image, image, mask=mask)

    return np.where(placement[:, :, 0] != 0)


def hist_match(source, template):
    old_shape = source.shape
    source = source.ravel()
    template = template.ravel()

    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(old_shape)


def blend(roi, class_img, flag=1):
    gray = cv2.cvtColor(class_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    if flag:
        h, w = mask.shape
        center = (w // 2, h // 2)

        # if width = 1, C++ error in seamlessClone
        dst = cv2.seamlessClone(class_img, roi, mask, center, cv2.NORMAL_CLONE)
        output = cv2.addWeighted(dst, 0.3, class_img, 0.7, 0)
        output[np.where(mask_inv != 0)] = roi[np.where(mask_inv != 0)]
        return output

    else:
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(class_img, class_img, mask=mask)
        dst = bg + fg
        return dst


def smooth_edges(dst, mask):
    temp = dst.copy()
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    edge = cv2.Canny(mask, 50, 150, 3)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edge, kernel, iterations=1)
    blurred = cv2.GaussianBlur(temp, (5, 5), 0)
    dst[np.where(dilated != 0)] = blurred[np.where(dilated != 0)]


def get_iou(bb1, bb2):
    x1, y1, x2, y2 = bb1
    a1, b1, a2, b2 = bb2

    x_left = max(x1, a1)
    y_top = max(y1, b1)
    x_right = min(x2, a2)
    y_bot = min(y2, b2)

    if x_right < x_left or y_bot < y_top:
        return 0.0

    intersection_area = (x_right - x_left + 1) * (y_bot - y_top + 1)

    bb1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bb2_area = (a2 - a1 + 1) * (b2 - b1 + 1)

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou


def get_class_pos(label, class_id):
    mask = cv2.inRange(label, class_id, class_id)
    contours, _ = cv2.findContours(mask, 1, 2)

    class_placement = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        class_placement.append([x, y, x + w, y + h, 1])

    deletions = []
    for x1, y1, x2, y2, _ in class_placement:
        for a, b, c, d, _ in class_placement:
            if x1 > a and y1 > b and x2 < c and y2 < d:
                deletions.append([x1, y1, x2, y2, 1])
                break

    class_placement = [x for x in class_placement if x not in deletions]

    return class_placement


def viz_scaling_triangle(aug_obj):
    triangle = np.array([[0, aug_obj.rows], [aug_obj.cols / 2, aug_obj.horizon_line],
                         [aug_obj.cols, aug_obj.rows]], np.int32)
    temp = aug_obj.image.copy()
    cv2.fillConvexPoly(temp, triangle, (255, 255, 0))
    cv2.addWeighted(temp, 0.3, aug_obj.image, 0.7, 0, temp)

    return temp


def viz_placement(aug_obj):
    temp = aug_obj.image.copy()
    color = np.random.choice(range(256), size=3)
    color = tuple([int(x) for x in color])
    [cv2.circle(temp, (x, y), 5, color) for x, y in zip(aug_obj.col_value, aug_obj.row_value)]

    return temp
