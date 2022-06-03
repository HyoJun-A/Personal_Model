import dlib
import cv2
import os



# os.mkdir("/김고은2")

detector = dlib.get_frontal_face_detector() 
path ="D:/project/Personal_model/김고은/"
os.chdir(path)
files = os.listdir(path)
jpg_img = []
for file in files:
    if '.jpg' in file: 
        f = cv2.imread(file)
        jpg_img.append(f)
# img = cv2.imread("input2.jpg")


for i in jpg_img:
    faces = detector(i)
    print("{} faces are detected.".format(len(faces)))
    for face in faces:
        print("left, top, right, bottom : ",face.left(), face.top(), face.right(), face.bottom())
        cv2.rectangle(i,(face.left(), face.top()), (face.right(), face.bottom()), (0,0,255),2)
        
    win = dlib.image_window()
    win.set_image(i)
    win.add_overlay(faces)
    #얼굴인식한 output 저장
    cv2.imwrite("output2.jpg",i)

    #얼굴 부분만 따로 저장

    
    crop = i[face.top():face.bottom(),face.left():face.right()]
    cv2.imwrite("김고은2/cropped2.jpg",crop)