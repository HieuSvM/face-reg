import os
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder
#########test
model = load_model('model.h5')
name = []
for f in os.listdir('./training_dataset'):
    name.append(f)
names_encode = LabelEncoder().fit(name)
Y = names_encode.transform(name).tolist() #[0,1,2,3,4,5,6,7]
x=[]
x_test=[]
for f in os.listdir('./training_dataset'):
    dem=0
    for i in os.listdir('./training_dataset/' + f + '/'):
        name_tuple = os.path.splitext(i)
        if (('.txt' in name_tuple[1]))&('name' not in name_tuple[0]):
            if (dem <= (len(os.listdir('./training_dataset/' + f + '/'))-1) * 0.7):
                dem = dem + 1
                continue
            else:
                k = open('./training_dataset/' + f + '/' + i, "r")
                line = k.read()
                line = line[0:-1]
                inner_list = [elt.strip() for elt in line.split(',')]
                k.close()
                x.append([float(j) for j in inner_list])
                a = model.predict(np.array(x))
                ind = np.argsort(a[0])
                for b in range(len(a[0])):
                    if a[0][b]==a[0][ind[-1]]:
                        if ((a[0][ind[-1]])*100>95):
                            print("Ảnh "+ name_tuple[0] + " là: "+ name[b] + " với " + str((a[0][ind[-1]])*100) +"%")
                        else:
                            print(
                                "Ảnh " + name_tuple[0] + " là: " + "Unknown" + " với " + str((a[0][ind[-1]])*100) +"%")
                x.clear()
        dem = dem + 1

######Kiem tra file train file test
# y_train=[]
# x_train=[]
# x=[]
# x1=[]
# x_test=[]
# y_test=[]
# y_trainh=[]
# y_testh=[]
# dem1=0
# yt_la =[]
# yt_la1 =[]
# for f in os.listdir('./training_dataset'):
#     dem=0
#     for i in os.listdir('./training_dataset/' + f + '/'):
#         name_tuple = os.path.splitext(i)
#         if (('.txt' in name_tuple[1]))&('name' not in name_tuple[0]):
#             if(dem<=(len(os.listdir('./training_dataset/' + f + '/'))-1)*0.7):
#
#                 y_train.append(f)
#                 y_trainh.append(dem1)
#                 yt_la.append(name_tuple[0])
#             else:
#                 y_test.append(f)
#                 y_testh.append(dem1)
#                 yt_la1.append(name_tuple[0])
#             k= open('./training_dataset/' + f + '/'+i, "r")
#             line = k.read()
#             line = line[0:-1]
#             inner_list = [elt.strip() for elt in line.split(',')]
#             k.close()
#             if (dem <= (len(os.listdir('./training_dataset/' + f + '/'))-1) * 0.7):
#                 x.append([float(j) for j in inner_list])
#             else:
#                 x1.append([float(j) for j in inner_list])
#         dem = dem + 1
#     dem1 = dem1 + 1
# print(yt_la)
# print(yt_la1)