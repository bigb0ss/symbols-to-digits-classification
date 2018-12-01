import cv2
import numpy as np
import keras
import pickle

model=pickle.load(open('digits.pkl','rb'))

cam=cv2.VideoCapture(0)

while True:
	ret,frame=cam.read()

	hsv= cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	low=np.array([0,48,80])
	high=np.array([20,255,255])

	mask=cv2.inRange(hsv,low,high)

	frame=cv2.bitwise_and(frame,frame,mask=mask)

	img=np.array(cv2.resize(frame,(50,50)))
	img=img.reshape(1,50,50,3)

	res=model.predict(img)
	if np.argmax(res) == 0:
		cv2.putText(frame,"Detected : 1",(10,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

	if np.argmax(res)==1:
		cv2.putText(frame,"Detected : 2",(10,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

	if np.argmax(res)==2:
		cv2.putText(frame,"Detected : 3",(10,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

	if np.argmax(res)==3:
		cv2.putText(frame,"Detected : 4",(10,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

	if np.argmax(res)==4:
		cv2.putText(frame,"Detected : 5",(10,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)


	cv2.imshow("Output Window",frame)
	if cv2.waitKey(1)==27:
		break

cv2.destroyAllWindows()
cam.release()
print("Program terminated....\n")
