import cv2
from Augmenter import sign_augmenter as SA
import dataset_reader as DR

img_path = "/home/opletts/Stuff/Cityscapes/leftImg8bit/val"
lbl_path = "/home/opletts/Stuff/Cityscapes/gtFine/val"

data = DR.DatasetReader(image_path=img_path, label_path=lbl_path)

for i in xrange(len(data)):
    image, label = data[i]
    image = cv2.resize(image, (1024, 512))
    label = cv2.resize(label, (1024, 512))
    aug = SA.SignAugmenter(image, label)
    img = aug.place_sign(1)

    cv2.imshow("image", img)
    cv2.waitKey(0)
