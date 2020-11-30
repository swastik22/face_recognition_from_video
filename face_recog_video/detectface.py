from cv2 import cv2
import numpy as np 
import face_recognition
import os

path='dataset'
images=[]
classNames=[]
mylist=os.listdir(path)
print(mylist)
for cl in mylist:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist
encodelistknown=findEncodings(images)
print('Encoding Completed')



cap=cv2.VideoCapture('input1.mp4')
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    check, frame = cap.read()
    #imgS=cv2.resize(frame,(0,0),None,0.25,0.25)
    #imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray, 1.3, 5)
    #facesCurFrame=face_recognition.face_locations(frame)
    encodeCurFrame=face_recognition.face_encodings(frame)
    for encodeFace in encodeCurFrame:
        matches=face_recognition.compare_faces(encodelistknown,encodeFace)
        faceDis= face_recognition.face_distance(encodelistknown,encodeFace)
        matchIndex=np.argmin(faceDis)
        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            for x,y,w,h in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
                cv2.putText(frame,name,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        


    '''    
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.putText(frame,name,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    '''

    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()