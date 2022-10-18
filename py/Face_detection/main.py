import cv2 as cv

cap = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

while(True):
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors= 5,
        minSize = (30, 30),
    )


    for (x, y ,w ,h) in faces:
        cv.rectangle(frame , (x,y), (x+w, y+h), (0,255, 0 ),2)

    cv.imshow('frame', frame)
 
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()    
cv.destroyAllWindows()

print("Code Complete")