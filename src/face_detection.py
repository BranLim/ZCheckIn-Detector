from imutils import paths
import face_recognition
import imutils
import pickle
import cv2
import os
import sys


imagesFolder = "faces"
knownEncoding = []
knownNames = []

face_dataset = "registered_faces.pickle"
cascPath = "haarcascade_frontalface_default.xml"
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

imagePaths = list(paths.list_images(imagesFolder))
num_of_image = len(imagePaths)

if num_of_image == 0:
    print("nothing to process")

elif num_of_image > 0:
    
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

    print("[INFO] serialising encodings...")
    data = {"encoding": knownEncoding, "names": knownNames}

    f = open(face_dataset, 'wb')
    f.write(pickle.dumps(data))
    f.close()
    print("[INFO] serialising encodings done")
