import dlib
import os
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
# load images
# detection
def color_histogram_of_training_image(img):
    if 'Obama' in img:
        data_source = 'Obama'
    elif 'Trump' in img:
        data_source = 'Trump'
    elif 'Putin' in img:
        data_source = 'Putin'
    elif 'Biden' in img:
        data_source = 'Biden'
    elif 'MTP' in img:
        data_source = 'MTP'
    elif 'Jack' in img:
        data_source = 'Jack'
    elif 'Bac' in img:
        data_source = 'Bac'
    img = dlib.load_rgb_image(img)
    img1_detection = detector(img, 1)
    img1_shape = sp(img, img1_detection[0])
    #khuon mat thu face trong anh
    img1_aligned = dlib.get_face_chip(img, img1_shape)
    #bien doi thanh vector 128
    img1_representation = facerec.compute_face_descriptor(img1_aligned)
    feature_data=''
    for i in range(0, 128):
        feature_data += str(img1_representation[i]) + ','
    with open('training.csv', 'a') as myfile:
        myfile.write(feature_data + data_source + '\n')

def training():

    # red color training images
    for f in os.listdir('./training_dataset/Obama'):
        color_histogram_of_training_image('./training_dataset/Obama/' + f)

    # yellow color training images
    for f in os.listdir('./training_dataset/Trump'):
        color_histogram_of_training_image('./training_dataset/Trump/' + f)

    # green color training images
    for f in os.listdir('./training_dataset/Putin'):
        color_histogram_of_training_image('./training_dataset/Putin/' + f)

    # orange color training images
    for f in os.listdir('./training_dataset/Biden'):
        color_histogram_of_training_image('./training_dataset/Biden/' + f)

    # white color training images
    for f in os.listdir('./training_dataset/MTP'):
        color_histogram_of_training_image('./training_dataset/MTP/' + f)

    # black color training images
    for f in os.listdir('./training_dataset/Jack'):
        color_histogram_of_training_image('./training_dataset/Jack/' + f)

    # brown color training images
    for f in os.listdir('./training_dataset/Bac'):
        color_histogram_of_training_image('./training_dataset/Bac/' + f)
training()