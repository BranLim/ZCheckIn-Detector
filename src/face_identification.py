from imutils.video import VideoStream
from imutils.video import FPS
from datetime import datetime
from datetime import timedelta
import imutils
import face_recognition
import pickle
import cv2
import time
import os
import sys
#import pdb
import csv
try:
    import temp_reader
except:
    pass

current_app_path = os.path.abspath(os.path.dirname(__file__))

check_in_entry_datetime_format = "%d-%m-%Y %H:%M:%S"

cascPath = os.path.join(current_app_path,"../models/haarcascade_frontalface_default.xml")
face_dataset =  os.path.join(current_app_path,"../data/registered_faces.pickle")
temperature_folder = os.path.join(current_app_path,"../data/temperature")

rgb_green = (0,255,0)
rgb_red = (0,0,255)

face_data = {}
check_in_record = {}

detector = cv2.CascadeClassifier(cascPath)

def get_record_file():
    return f'{temperature_folder}/{datetime.now(tz=None).strftime("%Y-%m-%d")}.csv'

def read_human_temperature():
    if 'temp_reader' not in sys.modules:
        return 0.0
    return temp_reader.read_temperature()

def create_record_file():
    if not os.path.exists(temperature_folder):
        os.makedirs(temperature_folder)
    
    record_file = get_record_file()
    if not os.path.exists(record_file):
        with open(record_file, 'w') : pass
    
    return record_file

def read_records(record_file):
    existing_records = {}

    if not os.path.exists(record_file):
        return existing_records
    
    with open(record_file, 'r') as csvfile:
        recordReader = csv.reader(csvfile, delimiter=',')
        for row in recordReader:
            check_in_entry = []

            try:
                check_in_entry.append((row[1], row[2]))
            except:
                print("no record")
                pass
            try:
                check_in_entry.append((row[3], row[4]))
            except:
                print("no record")
                pass

            existing_records[row[0]] = check_in_entry
    return existing_records
                

def write_record(record_file, records):     

    with open(record_file, 'w') as csvfile:
       recordWriter = csv.writer(csvfile, delimiter=',')
       for name, record in records.items():        
            temperatures = [entry for row in record for entry in row]
            csvRow = [name]
            csvRow.extend(temperatures)                
            recordWriter.writerow(csvRow)


def check_in(name, human_temperature, check_in_time):

    can_append_entry = False
   
    record_file = create_record_file()    
    
    global check_in_record
    user_entry = check_in_record.get(name)

    formatted_date_time = check_in_time.strftime(check_in_entry_datetime_format)
    entry = (formatted_date_time, human_temperature)
    
    if user_entry is None or not user_entry:
        user_entry = []
        check_in_record[name] = user_entry
        can_append_entry = True       
        
    elif len(user_entry) == 1:        
        check_time = datetime.strptime(user_entry[0][0],check_in_entry_datetime_format)
        if datetime.now(tz=None) - check_time >= timedelta(hours=4):
            can_append_entry = True

    if can_append_entry:
        user_entry.append(entry)
        print(f"{name} checked in at {formatted_date_time}")
        write_record(record_file, check_in_record)   


def identify_face(original_frame, rgb_frame, gray_frame):

    rects = detector.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))

    boxes = [(y, x+w, y+h, x) for (x, y, w, h) in rects]    
    num_faces = len(boxes)

    user_temperature = 0.0

    if num_faces > 1:
        cv2.putText(original_frame, "Error: Multiple faces detected.", (40,40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2 )
    elif num_faces == 1:
        live_encodings = face_recognition.face_encodings(rgb_frame, boxes)

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
            cv2.rectangle(original_frame, (left, top), (right, bottom), box_colour, 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(original_frame, f'{name} (temp: {user_temperature})', (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_colour, 2)


def init_face_data():
    if os.path.exists(face_dataset):
        global face_data
        face_data = pickle.loads(open(face_dataset, "rb").read())
    

def init_facial_recognition_feed():

    print("[INFO] starting video stream...")

    vs = VideoStream(src=2).start()

    time.sleep(2.0)
    fps = FPS().start()    

    while True:
        
        frame = vs.read()
        frame = imutils.resize(frame, width=480)      
 
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        identify_face(original_frame=frame, rgb_frame=rgb2, gray_frame=gray2)

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
    check_in_record = read_records(record_file = get_record_file())    
    init_facial_recognition_feed()