import numpy as np 
import cv2 as cv

cap = cv.VideoCapture(0)
#face_cascade = cv.CascadeClassifier('cv.data.haarcascades' + 'haarcascade_frontalface_default.xml')
#eye_cascade = cv.CascadeClassifier('cv.data.haarcascades' + 'haarcascade_eye.xml')
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
prof_cascade = cv.CascadeClassifier('haarcascade_profileface.xml')

while True:
    ret, frame = cap.read()

    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.3, 5) #will not pick up face profiles, just front

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x,y), (x+w, y+h), (235,235,0), 2)
        cv.putText(frame, 'face', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 0), 1)

        # Little bit faster because we are only searching face area for eyes.
        # Downside is that once face detection breaks, eye detection also breaks!!

        roi_grey = grey[y:y+h, x:x+w] #ys first for rows here
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_grey, 1.3, 5)

        for (ex, ey, ew, eh) in eyes:
            cv.circle(roi_color, (int((2*ex+ew)/2), int((2*ey+eh)/2)), int(((2*ex+ew)/2)-ex), (0,0,255), 2)
            cv.putText(roi_color, 'eye', (ex, ey-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # backup eye detector!
    eyes2 = eye_cascade.detectMultiScale(grey, 1.3, 5)
    for (ex, ey, ew, eh) in eyes2:
        cv.circle(frame, (int((2*ex+ew)/2), int((2*ey+eh)/2)), int(((2*ex+ew)/2)-ex), (180,120,120), 2)
        cv.putText(frame, 'eye', (ex, ey-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (180,120,120), 1)   

    #add in face profile detector
    # interesting that profile detector only seems to work in one direction!
    profs = prof_cascade.detectMultiScale(grey, 1.3, 5)
    for (px, py, pw, ph) in profs:
        cv.rectangle(frame, (px, py), (px+pw,py+ph), (120, 235, 0), 2)


    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()