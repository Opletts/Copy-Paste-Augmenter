import os
import cv2


class DatasetReader:
	"""Class built to read images and its corresponding pixel-wise segmented labels."""

	def __init__(self, image_path, label_path, shape=None, upsample=False):
		self.image_path = image_path
		self.label_path = label_path
		self.shape = shape
		self.interp = cv2.INTER_LANCZOS4 if upsample else cv2.INTER_AREA

		self.image_list = sorted(os.listdir(image_path))
		self.label_list = sorted(os.listdir(label_path))
		self.total_images = len(self.image_list)

	def __getitem__(self, image_number):
		if self.shape == None:
			return [cv2.imread(os.path.join(self.image_path, self.image_list[image_number])),
					cv2.imread(os.path.join(self.label_path, self.label_list[image_number]))]
		else:
			return [cv2.resize(cv2.imread(os.path.join(self.image_path, self.image_list[image_number])),
						   self.shape, interpolation=self.interp),
					cv2.resize(cv2.imread(os.path.join(self.label_path, self.label_list[image_number])),
						   self.shape, interpolation=self.interp)]

	def __len__(self):
		return self.total_images

if __name__ == '__main__':
	img_path = raw_input("Enter path to images : ")
	lbl_path = raw_input("Enter path to labels : ")

	data = DatasetReader(image_path=img_path, label_path=lbl_path)

	for i in xrange(len(data)):
		image, label = data[i]

		cv2.imshow("Image", image)
		cv2.imshow("Label", label)
		cv2.waitKey(0)
