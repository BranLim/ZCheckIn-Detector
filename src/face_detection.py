from imutils import paths
import face_recognition
import imutils
import pickle
import cv2
import os
import sys

current_app_path = os.path.abspath(os.path.dirname(__file__))

face_dataset =  os.path.join(current_app_path,"../data/registered_faces.pickle")
cascPath = os.path.join(current_app_path,"../models/haarcascade_frontalface_default.xml")

imagesFolder = os.path.join(current_app_path,"../data/faces/new")
processedFolder = os.path.join(current_app_path, "../data/faces/processed")

knownEncoding = []
knownNames = []

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

imagePaths = list(paths.list_images(imagesFolder))
num_of_image = len(imagePaths)

def move_processed(source, destination):
    os.rename(source, destination)

def save_dataset(data):
    if data:
        with open(face_dataset, 'wb') as f:
            f.write(pickle.dumps(data))

def load_existing_dataset(file):
     if os.path.exists(file):
        
        face_data = pickle.loads(open(file, "rb").read())

        if face_data:
            encoding = face_data.get("encoding")
            names = face_data.get("names")

            global knownEncoding
            global knownNames

            if encoding:
                knownEncoding.extend(encoding)
            if names:
                knownNames.extend(names)

def init_processed_directory():
    try:
        if not os.path.exists(processedFolder):
            os.makedirs()
    except:
        print("Error creating folder for processed images")
    

def process_images():
    if num_of_image == 0:
        print("nothing to process")
        return None  

    load_existing_dataset(face_dataset)    
    print(imagePaths)
    for (i, imagePath) in enumerate(imagePaths):

        print("[INFO] processing image {}/{}".format(i+1, len(imagePaths)))

        name = imagePath.split(os.path.sep)[-2]

        print(f"Image: {imagePath}")
        print(f"Name: {name}")

        image = cv2.imread(imagePath)
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.4,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) > 1:
            # If an image is found to have multiple faces, we have to skip it.
            continue

        print("Found {0} faces!".format(len(faces)))

        faces_list = []

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            faces_list.append([y, x+w, y+h, x])

        encodings = face_recognition.face_encodings(rgb, faces_list)
        for encoding in encodings:
            knownEncoding.append(encoding)
            knownNames.append(name)

    data = {"encoding": knownEncoding, "names": knownNames}

    print("[INFO] serialising encodings...")        
    save_dataset(data)   
    print("[INFO] serialising encodings done")


if __name__ == '__main__':
    init_processed_directory()
    process_images()