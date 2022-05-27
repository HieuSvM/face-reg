import os
from keras.models import load_model
import numpy as np
import dlib
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
#########test
model = load_model('mask_detector.model')
img = dlib.load_rgb_image("C:\\Users\\hmn20\\Pictures\\Doan\\test\\img1.jpg")
img1_detection = detector(img, 1)
for i in img1_detection:
    img1_shape = sp(img, i)
    img1_aligned = dlib.get_face_chip(img, img1_shape)
    img1_aligned = cv2.resize(img1_aligned,(224,224))
    img1_aligned = img_to_array(img1_aligned)
    img1_aligned = preprocess_input(img1_aligned)
    img1_aligned = np.expand_dims(img1_aligned, axis=0)
    (mask, withoutMask) = model.predict(img1_aligned)[0]

    # determine the class label and color we'll use to draw
    # the bounding box and text
    label = "Mask" if mask > withoutMask else "No Mask"
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

    # include the probability in the label
    print("{}: {:.2f}%".format(label, max(mask, withoutMask) * 100))

    cv2.imshow('A', img)

cv2.waitKey()