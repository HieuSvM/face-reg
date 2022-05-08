import dlib
import os
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
# load images
# detection
def color_histogram_of_training_image(img):
    img = dlib.load_rgb_image(img)
    img1_detection = detector(img, 1)
    if(len(img1_detection)==1):
        img1_shape = sp(img, img1_detection[0])
        #khuon mat thu face trong anh
        img1_aligned = dlib.get_face_chip(img, img1_shape)
        #bien doi thanh vector 128
        img1_representation = facerec.compute_face_descriptor(img1_aligned)
        feature_data=''
        for i in range(0, 128):
            feature_data += str(img1_representation[i]) + ','
        with open('traininglfw.csv', 'a') as myfile:
            myfile.write(feature_data + f + '\n')

def training():
    global f
    for f in os.listdir('./lfw'):
        for i in os.listdir('./lfw/' + f + '/'):
            color_histogram_of_training_image('./lfw/' + f + '/'+i)
training()