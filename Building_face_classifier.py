import numpy as np
import cv2
import os

def dist(x1,x2):
  return (sum((x2-x1)**2))


def knn(train,test,k=5):
    val=[]
    m=train.shape[0]
    for i in range (0,m):
        ix=train[i,:-1]
        iy=train[i,-1]
        d=dist(test,ix)
        val.append([d,iy])
      

    val=sorted(val)
    val=val[:k]
    val=np.array(val)
    new=np.unique(val[:,1],return_counts=True)
    idx=new[1].argmax()
    pred=new[0][idx]
    return (pred)



#init camera

cap=cv2.VideoCapture(0)

#Loading Data

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_data=[]
dataset_path='./'
skip=0
labels=[]

class_id=0 #labels for file
names={} #mapping name with id


#Data preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id]=fx[:-4]
        data_item=np.load(dataset_path+fx)
        face_data.append(data_item)

        #create labels for class
        target=class_id*np.ones((data_item.shape[0],))

        class_id+=1
        labels.append(target)


face_dataset=np.concatenate(face_data,axis=0)
face_labels=np.concatenate(labels,axis=0).reshape((-1,1))

trainset=np.concatenate((face_dataset,face_labels),axis=1)


#testing

while True:
    ret,frame=cap.read()
    if ret==False:
        continue

    faces=face_cascade.detectMultiScale(frame,1.3,5)
    for face in faces:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        offset=10

        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))

        out=knn(trainset,face_section.flatten())

        pred_name=names[int (out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        
    cv2.imshow("faces",frame)
    key_pressed=cv2.waitKey(1) &0xFF
    if key_pressed==ord('q'):
        break
    


cap.release()
cv2.destroyAllWindows()    






