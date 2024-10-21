import cv2
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math   
from cvzone.ClassificationModule import Classifier
from collections import Counter
import pyperclip 
import os

root = tk.Tk()
root.title("Nhóm 4 - Nhận diện cử chỉ")
canvas = tk.Canvas(root, width=640, height=700)
canvas.pack()

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("models/mymodel.keras")

offset = 20
imgSize = 300 
counter = 0
key = cv2.waitKey(1)

# Load labels from the dataset folder names
labels = os.listdir('Dataset')  # Đường dẫn tới thư mục chứa các thư mục nhãn
labels.sort()  # Sắp xếp danh sách nhãn (nếu cần)

def on_key_press(event):
    if event.char == 's':
        print("i am working")     

list1 = []
add_list = []

def btn_event(who=0):
    if who != 0:
        if len(list1) <= 5: 
            list1.append(who)
        else:
            global lable_list
            lable_list = list1.copy()
            list1.clear()

def me(x):
    if x == 0:
        counter = Counter(lable_list)
        most_common_item, count = counter.most_common(1)[0]
        add_list.append(str(most_common_item))
        result = ''.join(add_list)
        
    if x == 1:
        add_list.clear()
        result = ""

    label1.config(text=f"{result}")

def copy_text():
    text_to_copy = label1.cget("text")
    pyperclip.copy(text_to_copy)
    label2.place(x=220, y=500)

def update():
    success, img = cap.read()
    imgCopy = img.copy()
    hands, img = detector.findHands(img)

    if success:
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.photo = photo  

        if hands:
            label.place_forget()

            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

            imgCropShape = imgCrop.shape
    
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                widthGape = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, widthGape:wCal+widthGape] = imgResize
                pridiction, index = classifier.getPrediction(imgWhite, draw=False)
                label_name = labels[index]  # Lấy tên nhãn từ danh sách labels
                print(pridiction, label_name)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                widthGape = math.ceil((imgSize - hCal) / 2)
                imgWhite[widthGape:hCal+widthGape, :] = imgResize
                pridiction, index = classifier.getPrediction(imgWhite, draw=False)
                label_name = labels[index]  # Lấy tên nhãn từ danh sách labels
                
            cv2.rectangle(imgCopy, (x-offset, y-offset-50), (x+offset+w, y-offset), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgCopy, label_name, (x, y-30), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 4)
            cv2.rectangle(imgCopy, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)
            btn_event(label_name)
          
            rgb_frame = cv2.cvtColor(imgCopy, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.photo = photo 
            root.bind('<KeyPress>', on_key_press)
            
        else: 
            label.place(x=220, y=200)

    root.after(5, update) 


label_group = tk.Label(root, text="NHÓM 4", font=("Helvetica", 22), fg="blue")
label_group.place(x=12, y=5)

label_name = tk.Label(root, text="XỬ LÝ ẢNH VÀ THỊ GIÁC MÁY TÍNH", font=("Helvetica", 18), fg="blue")
label_name.place(x=12, y=35)


image_ref = Image.open("image.png")  
image_ref = image_ref.resize((320, 320))  
photo_ref = ImageTk.PhotoImage(image_ref)

label_image = tk.Label(root, image=photo_ref)
label_image.place(x = 1111, y = 50)


      
label = tk.Label(root, text="Chưa tìm thấy", font=("Helvetica", 20), fg="red", height=2)
label2 = tk.Label(root, text="Đã sao chép", font=("Helvetica", 17), fg="blue")

label1 = tk.Label(root, text="", font=("Helvetica", 20), fg="white", height=1, bg='blue')
label1.place(x=12, y=545)

btn_add = tk.Button(root, text="Thêm", font=("Helvetica", 20), command=lambda:me(0), bd=4, width=8, justify="center", bg="blue", fg="white")
btn_add.place(x=323, y=600)

btn_clear = tk.Button(root, text="Xoá", font=("Helvetica", 20), command=lambda:me(1), bd=4, width=8, justify="center", bg="red", fg="white")
btn_clear.place(x=480, y=600)

btn_copy = tk.Button(root, text="Sao chép", font=("Helvetica", 20), command=lambda:copy_text(), bd=4, width=8, justify="center", bg="green", fg="white")
btn_copy.place(x=100, y=600)

update()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
