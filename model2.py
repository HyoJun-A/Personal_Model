import dlib
import cv2
import numpy as np

path = 'D:/project/Personal_Moder/김고은_moder'
img_name = 'input2.jpg'
full_path = path + '/' +img_name
    
img_array = np.fromfile(full_path, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

detector = dlib.get_frontal_face_detector() 
# img = cv2.imread("input2.jpg")
faces = detector(img)

print("{} faces are detected.".format(len(faces)))
for face in faces:
    print("left, top, right, bottom : ",face.left(), face.top(), face.right(), face.bottom())
    cv2.rectangle(img,(face.left(), face.top()), (face.right(), face.bottom()), (0,0,255),2)
    

    
win = dlib.image_window()
win.set_image(img)
win.add_overlay(faces)
#얼굴인식한 output 저장
cv2.imwrite(r"D:\project\Personal_Moder\test\output2.jpg",img)

#얼굴 부분만 따로 저장
crop = img[face.top():face.bottom(),face.left():face.right()]
cv2.imwrite(r"D:\project\Personal_Moder\test\cropped2.jpg",crop)