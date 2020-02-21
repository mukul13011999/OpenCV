import cv2
import numpy as np
def genrate_dataset(img, id, img_id ):
    cv2.imwrite("data/user."+str(id)+"."+str(img_id)+".jpg",img)


def draw_boundary(img,classifier , scaleFacter, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features  = classifier.detectMultiScale(gray_img,scaleFacter,minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img,(x,y), (x+w, y+h), color , 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords=[x, y, w, h]

    return coords


def detect(img, faceCascade,eyeCascade,body, img_id):
    color ={"blue":(255,0,0),"red":(0,0,255),"green":(0,255,0)}
    coords = draw_boundary(img , faceCascade, 1.1, 10, color['red'],"face")
    coords = draw_boundary(img, body, 1.1, 4, color['blue'], "body")

    if len(coords)==4:
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        user_id = 1
        genrate_dataset(roi_img, user_id, img_id)
        #coords = draw_boundary(img, faceCascade, 1.1, 14, color['red'], "face")
        coord = draw_boundary(roi_img, eyeCascade, 1.1, 14, color['blue'], "Eye")

    return img



cap = cv2.VideoCapture(0)
ret,frame1 = cap.read()
ret,frame2 = cap.read()
img_id = 0
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
body = cv2.CascadeClassifier("haarcascade_fullbody1.xml")
while cap.isOpened():

    diff = cv2.absdiff(frame1,frame2)
    gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _,thresh = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours,_ = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour)<700:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame1,"status: {}".format('movement'),(10 ,20),cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,0,255),3)
    #cv2.drawContours(frame1,contours,-1,(0,255,0),2)
    frame1 = detect(frame1, faceCascade, eyeCascade,body, img_id)
    cv2.imshow("vdo",frame1)
    frame1 =frame2
    ret,frame2 = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()