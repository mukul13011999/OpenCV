import cv2

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


def detect(img, faceCascade,eyeCascade,body,upperBody,img_id):
    color ={"blue":(255,0,0),"red":(0,0,255),"green":(0,255,0)}
    #coords = draw_boundary(img , body, 1.1, 2, color['blue'],"body")
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['red'], "face")
    #coords = draw_boundary(img, upperBody, 1.1, 2, color['green'], "upperBody")

    if len(coords)==4:
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        user_id = 1
        genrate_dataset(roi_img, user_id, img_id)
        #coord = draw_boundary(roi_img, eyeCascade, 1.1, 14, color['blue'], "Eye")
    return img


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
body = cv2.CascadeClassifier('haarcascade_fullbody1.xml')
#mouthCascade = cv2.CascadeClassifier("Mouth.xml")
upperBody= cv2.CascadeClassifier("haarcascade_upperbody2.xml")

video_capture =cv2.VideoCapture(0)
img_id = 0

while True:
    _,img = video_capture.read()
    img = detect(img, faceCascade,eyeCascade,body,upperBody,img_id)
    cv2.imshow("face detection", img)
    #img_id += 1
    if cv2.waitKey(1) &0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
