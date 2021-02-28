#user_checkin.py

from logging import handlers
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
import constants
# import pdb
import logging
from logging.handlers import TimedRotatingFileHandler
import csv
try:
    import temp_reader
except:
    pass

current_app_path = os.path.abspath(os.path.dirname(__file__))

logger = logging.getLogger("User Checkin")

temperature_folder = os.path.join(current_app_path, "../data/temperature")

rgb_green = (0, 255, 0)
rgb_red = (0, 0, 255)

face_data = {}
check_in_record = {}

def get_checkin_record_filename():
    return f'{datetime.now(tz=None).strftime("%Y-%m-%d")}.csv'


def read_human_temperature():
    if 'temp_reader' not in sys.modules:
        logger.info("Thermal sensor is not ready.")
        return 0.0
    return temp_reader.read_temperature()


def create_record_file():

    global temperature_folder

    if not os.path.exists(temperature_folder):
        os.makedirs(temperature_folder)
    
    record_file = os.path.join(temperature_folder, get_checkin_record_filename())
    if not os.path.exists(record_file):
        with open(record_file, 'w'):
            pass

    return record_file


def init_checkin_records(record_file):
    existing_records = {}

    if not os.path.exists(record_file):
        return existing_records

    with open(record_file, 'r') as csvfile:
        record_reader = csv.reader(csvfile, delimiter=',')
        for row in record_reader:
            check_in_entry = []

            try:
                check_in_entry.append((row[1], row[2]))
            except:
                pass
            try:
                check_in_entry.append((row[3], row[4]))
            except:
                pass

            existing_records[row[0]] = check_in_entry
    return existing_records


def write_record(record_file, records):
    logger.info("Saving checkin record to file...")
    with open(record_file, 'w') as csvfile:
        recordWriter = csv.writer(csvfile, delimiter=',')
        for name, record in records.items():
            temperatures = [entry for row in record for entry in row]
            csvRow = [name]
            csvRow.extend(temperatures)
            recordWriter.writerow(csvRow)
    logger.info("Checkin record saved.")


def check_in(name, human_temperature, check_in_time):

    can_append_entry = False

    # creates a checkin record file if it does not exists
    record_file = create_record_file()

    global check_in_record
    user_entry = check_in_record.get(name)

    formatted_date_time = check_in_time.strftime(
        constants.CHECK_IN_DATETIME_FORMAT)
    entry = (formatted_date_time, human_temperature)

    if user_entry is None or not user_entry:
        user_entry = []
        check_in_record[name] = user_entry
        can_append_entry = True

    elif len(user_entry) == 1:
        check_time = datetime.strptime(
            user_entry[0][0], constants.CHECK_IN_DATETIME_FORMAT)
        if datetime.now(tz=None) - check_time >= timedelta(hours=4):
            logger.info("User have checked in before. Updating existing record...")
            can_append_entry = True

    if can_append_entry:
        user_entry.append(entry)
        logger.info(f"{name} checked in at {formatted_date_time} with temperature {human_temperature} degrees Celsius")
        write_record(record_file=record_file, records=check_in_record)


def identify_face(face_detector,original_frame, rgb_frame, gray_frame):

    
    detected_faces = face_detector.detectMultiScale(
        gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))

    detected_faces_boxes = [(y, x+w, y+h, x) for (x, y, w, h) in detected_faces]
    num_faces = len(detected_faces)

    user_temperature = 0.0

    if num_faces > 1:
        logger.warn("Detected multiple faces. Ignoring...")
        cv2.putText(original_frame, "Error: Multiple faces detected.",
                    (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        return None
    elif num_faces == 0:
        return None
    
    logger.info("Detected one face. Processing...")
    live_encodings = face_recognition.face_encodings(rgb_frame, detected_faces_boxes)

    current_time = datetime.now(tz=None)
    logger.info(f"Found {len(live_encodings)} encodings")

    '''
    Although most solutions will iterate through the live_encodings. But in our case, there should only be one encoding. So we will take the first one.
    '''

    encoding = live_encodings[0]

    user_name = "Unknown"

    if face_data:
        matches = face_recognition.compare_faces(
            face_data["encoding"], encoding, tolerance=0.4)

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                user_name = face_data["names"][i]
                counts[user_name] = counts.get(user_name, 0) + 1
            user_name = max(counts, key=counts.get)

    user_temperature = read_human_temperature()

    try:
        if user_name == "Unknown":
            pass

        else:
            logger.info("Face recognised. To check in user.")
            check_in(name=user_name, human_temperature=user_temperature,
                        check_in_time=current_time)
            logger.info("User completed checked in.")
    except:
        pass

    return (detected_faces_boxes, user_name, user_temperature)
        

def show_detection(original_frame,user_temperature, detected_faces, detected_names):

    global rgb_green
    global rgb_red

    box_colour = rgb_green
    if user_temperature >= 37.5:
        box_colour = rgb_red

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(detected_faces, detected_names):
        # draw the predicted face name on the image
        cv2.rectangle(original_frame, (left, top),
                        (right, bottom), box_colour, 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(original_frame, f'{name} (temp: {user_temperature})', (
            left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_colour, 2)


def load_registered_faces(face_name_data):
    if os.path.exists(face_name_data):
        global face_data
        face_data = pickle.loads(open(face_name_data, "rb").read())


def detect_and_process_faces(face_detection_model):

    detector = cv2.CascadeClassifier(face_detection_model)

    logger.info("Starting video stream...")    
    vs = VideoStream(src=2).start()

    logger.info("Warming up camera")
    time.sleep(2.0)
    fps = FPS().start()

    while True:

        frame = vs.read()
        frame = imutils.resize(frame, width=480)

        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detected_face_temperature = identify_face(face_detector=detector,original_frame=frame, rgb_frame=rgb_frame, gray_frame=grayscale_frame)

        if detected_face_temperature is not None:
            show_detection(original_frame=frame, user_temperature=detected_face_temperature[2], detected_faces=detected_face_temperature[0], detected_names=[detected_face_temperature[1]])   

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
    logger.info("Elapsed time: {:.2f}".format(fps.elapsed()))
    logger.info("Approx. FPS: {:.2f}".format(fps.fps()))

    logger.info("Stopping video stream and closing window")
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


def setup_logging():
    global logger

    log_folder = os.path.join(current_app_path,f'../{constants.LOG_FOLDER}')
    try:        
        if not os.path.exists(log_folder):
            os.mkdir(log_folder)
    except:
        print("[ERROR] Cannot create logging directory")
    
    log_file = os.path.join(log_folder, constants.USER_CHECKIN_LOG)

    logging_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s","%Y-%m-%d %H:%M:%S")
    logger.setLevel(logging.INFO)

    time_rotating_log_file_handler = TimedRotatingFileHandler(log_file, when="d", interval=1, backupCount=5)
    time_rotating_log_file_handler.setFormatter(logging_formatter)

    logger.addHandler(time_rotating_log_file_handler)


if __name__ == '__main__':

    setup_logging()

    haar_face_classifier = os.path.join(
    current_app_path, "../models/haarcascade_frontalface_default.xml")
    face_name_mapping = os.path.join(
        current_app_path, "../data/registered_faces.pickle")

    logger.info("App started.")

    logger.info("Loading registered faces...")
    load_registered_faces(face_name_data=face_name_mapping)
    logger.info("Loaded registered faces.")

    logger.info("Initialising checkin database.")
    check_in_record = init_checkin_records(record_file=get_checkin_record_filename())
    logger.info("Checkin database ready.")

    logger.info("Start facial recognition...")
    detect_and_process_faces(face_detection_model=haar_face_classifier)

    logger.info("App shutdown")
