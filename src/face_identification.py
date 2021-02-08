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
from datetime import timedelta
import uuid
import requests
import pdb
from collections import Counter
import temp_reader

check_in_entry_datetime_format = "%d-%m-%y %H:%M:%S"
current_app_path = os.path.abspath(os.path.dirname(__file__))
cascPath = os.path.join(current_app_path,"../models/haarcascade_frontalface_default.xml")
face_dataset =  os.path.join(current_app_path,"../data/registered_faces.pickle")
rgb_green = (0,255,0)
rgb_red = (255,0,0)

face_data = {}
unknown_faces = {}
check_in_record = {}

detector = cv2.CascadeClassifier(cascPath)

def export_check_in_to_csv():
    pass

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def hash_image(image, hashSize = 8):
    resized = cv2.resize(image, (hashSize+1, hashSize))

    diff = resized[:, 1:] > resized[:, :-1]

    return sum([2 ** i for (i,v) in enumerate(diff.flatten()) if v])

def read_human_temperature():
    return temp_reader.read_temperature()

def check_in(name, human_temperature, check_in_time):

    can_append_entry = False
    formatted_date_time = check_in_time.strftime(check_in_entry_datetime_format)
    formatted_date = check_in_time.strftime("%d-%m-%y")
    
    global check_in_record

    records = check_in_record.get(formatted_date)
    entry = (formatted_date_time, name, human_temperature)
    
    if records is None or not records:
        records = []
        can_append_entry = True
        check_in_record[formatted_date] = records
        
    else:
        for record in records:
            check_time = datetime.strptime(record[0],check_in_entry_datetime_format)
            if (name not in record) or (name in record and datetime.now(tz=None) - check_time >= timedelta(hours=4) and Counter(elem[1] for elem in records)[name] < 2):
                can_append_entry = True
                break

    if can_append_entry:
        records.append(entry)
        print(f"{name} checked in at {formatted_date_time}")

    

def upload_user(current_frame, image_id , region, current_time):
    
    print('Saving region to file.')
    cv2.imwrite(os.path.join(current_app_path, f"{image_id}.png"), region)

def init_face_data():
    if os.path.exists(face_dataset):
        global face_data
        face_data = pickle.loads(open(face_dataset, "rb").read())
    

def init_facial_recognition_feed():

    print("[INFO] starting video stream...")

    vs = VideoStream(src=0).start()

    time.sleep(2.0)
    fps = FPS().start()

    global unknown_faces

    while True:
        
        frame = vs.read()
        frame = imutils.resize(frame, width=800)

        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rects = detector.detectMultiScale(gray2, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))

        boxes = [(y, x+w, y+h, x) for (x, y, w, h) in rects]    
        num_faces = len(boxes)

        user_temperature = 0.0

        if num_faces > 1:
            cv2.putText(frame, "Error: Multiple faces detected.", (40,40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2 )
        elif num_faces == 1:
            live_encodings = face_recognition.face_encodings(rgb2, boxes)

            names = []
            current_time = datetime.now(tz=None)

            for encoding in live_encodings:
                
                name = "Unknown"

                if face_data:
                    matches = face_recognition.compare_faces(face_data["encoding"], encoding, tolerance=0.4)

                    if True in matches:
                        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                        counts = {}

                        for i in matchedIdxs:
                            name = face_data["names"][i]
                            counts[name] = counts.get(name, 0) + 1
                        name = max(counts, key=counts.get)

                user_temperature = read_human_temperature()

                try:
                    if name == "Unknown":
                        pass
                        # Get the first tuple (y, width, height, x ) in the list of boxes
                        #ROI = frame[boxes[0][0]:boxes[0][2],boxes[0][3]:boxes[0][1]]
                        #roi_to_hash = to_grayscale(frame)
                        #image_hash = hash_image(roi_to_hash)
                        
                        #detected_image = unknown_faces.get(image_hash)
                        #if detected_image is None or not detected_image:
                          #  image_uuid = uuid.uuid4().hex
                          #  unknown_faces[image_hash] = (image_uuid, ROI)
                            #upload_user(current_frame=frame,image_id = image_uuid,region=ROI, current_time=current_time)
                        #check_in(user_uuid, human_temperature=user_temperature, check_in_time=current_time)
                        
                    else:
                        check_in(name, human_temperature = user_temperature, check_in_time=current_time)
                except:
                    pass

                names.append(name)

            box_colour = rgb_green
            if user_temperature >= 37.5:
                box_colour = rgb_red
                
            # loop over the recognized faces
            for ((top, right, bottom, left), name) in zip(boxes, names):
                # draw the predicted face name on the image
                cv2.rectangle(frame, (left, top), (right, bottom), box_colour, 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_colour, 2)


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
    init_face_data()
    init_facial_recognition_feed()