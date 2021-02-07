from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import face_recognition
import pickle
import cv2
import time
import os
import sys
from datetime import datetime


current_app_path = os.path.abspath(os.path.dirname(__file__))
cascPath = os.path.join(current_app_path,"../models/haarcascade_frontalface_default.xml")
face_dataset =  os.path.join(current_app_path,"../check-in/registered_faces.pickle")

face_data = pickle.loads(open(face_dataset, "rb").read())

detector = cv2.CascadeClassifier(cascPath)

def read_human_temperature():
    pass

def check_in(name, human_temperature):
    current_time = datetime.now(tz=None)
    
    pass


def upload_user():
    pass


def init_facial_recognition_feed():

    print("[INFO] starting video stream...")

    vs = VideoStream(src=2).start()

    time.sleep(2.0)
    fps = FPS().start()

    while True:
        
        frame = vs.read()
        frame = imutils.resize(frame, width=800)

        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rects = detector.detectMultiScale(gray2, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        boxes = [(y, x+w, y+h, x) for (x, y, w, h) in rects]    
        num_faces = len(boxes)

        if num_faces > 1:
            cv2.putText(frame, "Error: Multiple faces detected.", (40,40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2 )
        else:
            live_encodings = face_recognition.face_encodings(rgb2, boxes)

            names = []

            for encoding in live_encodings:
                matches = face_recognition.compare_faces(face_data["encoding"], encoding, tolerance=0.45)
                name = "Unknown"

                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}

                    for i in matchedIdxs:
                        name = face_data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)

                user_temperature = read_human_temperature()

                if name == "Unknown":
                    upload_user()
                    check_in(name, human_temperature=user_temperature)
                    pass
                else:
                    check_in(name, human_temperature = user_temperature)

                names.append(name)

            # loop over the recognized faces
            for ((top, right, bottom, left), name) in zip(boxes, names):
                # draw the predicted face name on the image
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


        # display the image to our screen
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':
    init_facial_recognition_feed()