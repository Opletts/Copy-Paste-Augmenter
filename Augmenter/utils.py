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

def blend(roi, sign):
    gray = cv2.cvtColor(sign, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    sign = hist_match(sign, roi)

    bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    fg = cv2.bitwise_and(sign, sign, mask=mask)
    dst = bg + fg

    return dst
