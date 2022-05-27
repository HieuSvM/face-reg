import cv2
import pandas as pd
import math
import PIL.ImageTk, PIL.Image
from tkinter import *
from tkinter import filedialog
from threading import Thread
import sys
import dlib
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
window=Tk()
Distance = []
Dis = []
Outold = []
outnew = []
filename = None
photo = None
frame = Label(window)
def moanh():
    global frame,photo,filename,az,az1
    filename = filedialog.askopenfilename(title='open')
    inp = cv2.imread(filename)
    scale_percent = 60  # percent of original size
    width = int(inp.shape[1] * scale_percent / 100)
    height = int(inp.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    inp = cv2.resize(inp, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite("input.jpg", inp)
    photo = dlib.load_rgb_image("input.jpg")
    try:
        az = PIL.Image.open("input.jpg")
        az = PIL.ImageTk.PhotoImage(az)
        frame.configure(image=az)
        frame.image = az
        az1 = cv2.imread("input.jpg")
    except:
        pass
# Đọc tệp csv với pandas và đặt tên cho từng cột
index = []
for i in range(0,129):
    index.append(str(i))
csv = pd.read_csv('training.csv', names=index, header=None)
csv.to_string()
dataa = IntVar()
def getColorName():
    global text, ra
    try:
        img1_detection = detector(photo, 1)
        for face in img1_detection:
            img1_shape = sp(photo, face)
            x = face.left()
            y = face.top()
            width = face.right() - x
            height = face.bottom() - y
            # ve hcn
            # khuon mat thu face trong anh
            img1_aligned = dlib.get_face_chip(photo, img1_shape)
            # bien doi thanh vector 128
            img1_representation = facerec.compute_face_descriptor(img1_aligned)
            for m in range(len(csv)):
                d=0
                for v in range(0,128):
                    d += (img1_representation[v] - float(csv.loc[m, str(v)]))*(img1_representation[v] - float(csv.loc[m, str(v)]))# loc: nhan các hàng hoặc cột với cụ thể nhãn từ chỉ mục
                d = math.sqrt(d)
                Distance.append(d)
                cname = csv.loc[m, str(128)]
                Outold.append(cname)
            for i in range(0, len(Distance) - 1):
                for j in range(i + 1, len(Distance)):
                    if (Distance[i] > Distance[j]):
                        tmp = Distance[i]
                        Distance[i] = Distance[j]
                        Distance[j] = tmp
                        tg = Outold[i]
                        Outold[i] = Outold[j]
                        Outold[j] = tg
            k = dataa.get()
            print(Distance)
            n = 0
            nguong = 0.35
            for l in range(k):
                if (Distance[l]<nguong):
                    Dis.append(Distance[l])
                    outnew.append(Outold[l])
            if (len(outnew)!=0):
                for f in range(len(outnew)):
                    u = outnew.count(outnew[f])
                    if (u > n):
                        n = u
                        ra = outnew[f]
                cv2.rectangle(az1,(x,y),(x+width,y+height),color=(255,0,0),thickness=2)
                if (width<100):
                    cv2.putText(az1,ra, (x,y+height-5), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(az1, ra, (x,y + height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                ra = "Unhnow"
                cv2.rectangle(az1, (x , y), (x + width, y + height), (255, 0, 0), 2)
                if (width < 100):
                    cv2.putText(az1, "Unknow", (x, y +height-5 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(az1, "Unknow", (x, y + height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            print (ra)
            del Distance[:]
            del Outold[:]
            del outnew[:]
            del Dis[:]
        cv2.imwrite("output.jpg",az1)
        ck = PIL.ImageTk.PhotoImage(PIL.Image.open("output.jpg"))
        frame.configure(image=ck)
        frame.image = ck
    except:
        sys.exit()
    finally:
        sys.exit()
def main():
    thred = Thread(target=getColorName)
    thred.start()
def out():
    sys.exit()
Button(window, text ="Chọn ảnh",width=10, command = moanh).pack(side=TOP)
Button(window, text ="Bắt đâu",width=10, command = main).pack(side=TOP)
Button(window, text ="Thoát",width=10, command = out).pack(side=TOP)
Label(window, text="Nhập K", font=("Arial Bold", 10)).pack(side=TOP)
label = Label(window, text="", font=("Arial Bold", 30))
entry = Entry(window,textvariable=dataa)
entry.pack(side=TOP)
frame.pack(side=TOP)
window.mainloop()