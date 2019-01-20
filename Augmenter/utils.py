import cv2
import numpy as np

def threshold(image, label, value):
    if type(value[0]) == int:
        mask = cv2.inRange(label, value, value)
        placement = cv2.bitwise_and(image, image, mask=mask)

        return np.where(placement[:, :, 0] != 0)

    else:
        mask = 0
        for i in value:
            mask += cv2.inRange(label, i, i)

        placement = cv2.bitwise_and(image, image, mask=mask)

        return np.where(placement[:, :, 0] != 0)

def hist_match(source, template):
	oldshape = source.shape
	source = source.ravel()
	template = template.ravel()

	s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
	t_values, t_counts = np.unique(template, return_counts=True)

	s_quantiles = np.cumsum(s_counts).astype(np.float64)
	s_quantiles /= s_quantiles[-1]
	t_quantiles = np.cumsum(t_counts).astype(np.float64)
	t_quantiles /= t_quantiles[-1]

	interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

	return interp_t_values[bin_idx].reshape(oldshape)

def blend(roi, class_img):
    gray = cv2.cvtColor(class_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    class_img = smooth_edges(class_img, mask)
    class_img = hist_match(class_img, roi)

    bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    fg = cv2.bitwise_and(class_img, class_img, mask=mask)
    dst = bg + fg

    return dst

def smooth_edges(dst, mask):
    temp = dst.copy()
    edge = cv2.Canny(mask, 50, 150, 3)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edge, kernel, iterations=1)
    blurred = cv2.GaussianBlur(temp, (7, 7), 0)
    dst[np.where(dilated != 0)] = blurred[np.where(dilated != 0)]

    return dst
