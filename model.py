import dlib
import cv2
import os
import numpy as np


# os.mkdir("/김고은_")

detector = dlib.get_frontal_face_detector() 
path ="D:/project/Personal_Moder/김고은/"
os.chdir(path)
files = os.listdir(path)
jpg_img = [] #숫자
jpg = []  #이름
for file in files:
    if '.jpg' in file: 
        f = cv2.imread(file)
        jpg_img.append(f)
        jpg.append(file)
# img = cv2.imread("input2.jpg")





for i in range(len(jpg_img)):
    
    faces = detector(jpg_img[i])
    print("{} faces are detected.".format(len(faces)))
    for face in faces:
        print("left, top, right, bottom : ",face.left(), face.top(), face.right(), face.bottom())
        cv2.rectangle(jpg_img[i],(face.left(), face.top()), (face.right(), face.bottom()), (0,0,255),2)
            
    win = dlib.image_window()
    win.set_image(jpg_img[i])
    win.add_overlay(faces)
        #얼굴인식한 output 저장
        # cv2.imwrite("output2.jpg",i)

        #얼굴 부분만 따로 저장
        
        # path = 'D:/project/Personal_Model/김고은_moder'
        # img_name = j
        # full_path = path + '/' +img_name
            
        # img_array = np.fromfile(full_path, np.uint8)
        # img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    j =str(i)
    crop = jpg_img[i][face.top():face.bottom(),face.left():face.right()]
    cv2.imwrite("D:/project/Personal_Moder/test/cropped"+j+".jpg",crop)