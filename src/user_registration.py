from imutils import paths
import face_recognition
import pickle
import cv2
import os
import shutil
import logging
from logging.handlers import TimedRotatingFileHandler
import constants

current_app_path = os.path.abspath(os.path.dirname(__file__))

logger = logging.getLogger("User Registration")

knownEncoding = []
knownNames = []


def move_processed_images(source_images, destination_folder_root):

    if type(source_images) is list:
        for imagePath in source_images:
            print(f'Images to move: {imagePath}')
            name = imagePath.split(os.path.sep)[-2]
            fileName = imagePath.split(os.path.sep)[-1]

            final_destination_folder = os.path.join(
                destination_folder_root, f'{name}')

            try:
                if not os.path.exists(final_destination_folder):
                    os.mkdir(path=final_destination_folder)
            except:
                pass
            else:
                final_destination_path = os.path.join(
                    final_destination_folder, fileName)
                print(f'Move Destination: {final_destination_path}')
                shutil.move(src=imagePath, dst=final_destination_path)


def save_registered_faces(registered_faces,face_name_mappings):
    if face_name_mappings:
        with open(registered_faces, 'wb') as f:
            f.write(pickle.dumps(face_name_mappings))


def load_registered_faces(face_name_mapping):
    if os.path.exists(face_name_mapping):

        face_data = pickle.loads(open(face_name_mapping, "rb").read())

        if face_data:
            encoding = face_data.get("encoding")
            names = face_data.get("names")

            global knownEncoding
            global knownNames

            if encoding:
                knownEncoding.extend(encoding)
            if names:
                knownNames.extend(names)


def init_processed_directory(processed_images_directory):
    try:
        if not os.path.exists(processed_images_directory):
            os.makedirs(processed_images_directory)
    except:
        print("Error creating folder for processed images")


def register_faces(image_paths, classifier):

    processed_images = []

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(classifier)

    global knownEncoding
    global knownNames

    # Goes through the image paths.
    for (i, image_path) in enumerate(image_paths):

        print("[INFO] processing image {}/{}".format(i+1, len(image_paths)))

        # The foldername is the name of the face
        name = image_path.split(os.path.sep)[-2]

        print(f"Image: {image_path}")
        print(f"Name: {name}")

        image = cv2.imread(image_path)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        detected_faces = faceCascade.detectMultiScale(
            grayscale_image, scaleFactor=1.4, minNeighbors=5, minSize=(30, 30))

        print("Found {0} faces!".format(len(detected_faces)))

        if len(detected_faces) > 1:
            # If an image is found to have multiple faces, we have to skip it. The images should be either selfies or profile pictures
            continue

        processed_images.append(image_path)

        faces_list = []

        # Draw a rectangle around the faces
        for (x, y, w, h) in detected_faces:
            #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            faces_list.append([y, x+w, y+h, x])

        encodings = face_recognition.face_encodings(rgb_image, faces_list)
        for encoding in encodings:
            knownEncoding.append(encoding)
            knownNames.append(name)

    return processed_images


def process_images(image_classifier, images_dir, processed_images_directory, registered_faces_database):

    imagePaths = list(paths.list_images(images_dir))
    num_of_image = len(imagePaths)

    if num_of_image == 0:
        logger.info("No new faces to process")
        return None

    logger.info("Processing images at: ")
    logger.info(f"{imagePaths}")

    processed_images = register_faces(
        image_paths=imagePaths, classifier=image_classifier)

    logger.info("Images processed.")

    logger.info("Moving processed images...")
    move_processed_images(source_images=processed_images,
                          destination_folder_root=processed_images_directory)

    logger.info("Processed images moved.")

    global knownEncoding
    global knownNames

    logger.info("Saving identified faces..")
    data = {"encoding": knownEncoding, "names": knownNames}
    save_registered_faces(registered_faces=registered_faces_database,face_name_mappings=data)
    logger.info("Identified faces saved.")

def setup_logging():
    global logger

    log_folder = os.path.join(current_app_path,f'../{constants.LOG_FOLDER}')
    try:        
        if not os.path.exists(log_folder):
            os.mkdir(log_folder)
    except:
        print("[ERROR] Cannot create logging directory")
    
    log_file = os.path.join(log_folder, constants.USER_REGISTRATION_LOG)

    logging_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s","%Y-%m-%d %H:%M:%S")
    logger.setLevel(logging.INFO)

    time_rotating_log_file_handler = TimedRotatingFileHandler(log_file, when="d", interval=1, backupCount=5)
    time_rotating_log_file_handler.setFormatter(logging_formatter)

    logger.addHandler(time_rotating_log_file_handler)



if __name__ == '__main__':

    setup_logging()

    registered_faces_database = os.path.join(
        current_app_path, "../data/registered_faces.pickle")
    haar_cascade_model_file = os.path.join(
        current_app_path, "../models/haarcascade_frontalface_default.xml")

    new_face_images_dir = os.path.join(current_app_path, "../data/faces/new")
    processed_face_images_dir = os.path.join(
        current_app_path, "../data/faces/processed")

    logger.info("App started.")

    logger.info("Initialising directory for processed images.")
    init_processed_directory(processed_face_images_dir)

    logger.info("Loading registered faces.")
    load_registered_faces(face_name_mapping=registered_faces_database)

    logger.info("Processing new faces...")
    process_images(image_classifier=haar_cascade_model_file, images_dir=new_face_images_dir,
                   processed_images_directory=processed_face_images_dir, registered_faces_database=registered_faces_database)
    logger.info("Finished processing new faces.")

    logger.info("App shutdown.")
