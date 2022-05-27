import dlib
import os
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
img = dlib.load_rgb_image("E:\\Desktop\\samsung\\Doan\\test\\img2.jpg")
img1_detection = detector(img, 1)
if (len(img1_detection) == 1):
    img1_shape = sp(img, img1_detection[0])
    img1_aligned = dlib.get_face_chip(img, img1_shape)
    img1_representation = facerec.compute_face_descriptor(img1_aligned)
    feature_data = ''
    for i in range(0, 127):
        feature_data += str(img1_representation[i]) + ','
    feature_data += str(img1_representation[i])
with open("E:\\Desktop\\samsung\\Doan\\test\\" + 'img2.txt', 'w') as myfile:
    myfile.write(feature_data)