from imutils import paths
import face_recognition
import pickle
import cv2
import os
import shutil

current_app_path = os.path.abspath(os.path.dirname(__file__))

knownEncoding = []
knownNames = []


def move_processed_images(source_images, destination_folder_root):
    if source_images is None:
        return None

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

            final_destination_path = os.path.join(
                destination_folder_root, f'{name}/{fileName}')
            print(f'Move Destination: {final_destination_path}')
            shutil.move(src=imagePath, dst=final_destination_path)


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
    for (i, imagePath) in enumerate(image_paths):

        print("[INFO] processing image {}/{}".format(i+1, len(imagePaths)))

        name = imagePath.split(os.path.sep)[-2]

        print(f"Image: {imagePath}")
        print(f"Name: {name}")

        image = cv2.imread(imagePath)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray, scaleFactor=1.4, minNeighbors=5, minSize=(30, 30))

        print("Found {0} faces!".format(len(faces)))

        if len(faces) > 1:
            # If an image is found to have multiple faces, we have to skip it. The images should be either selfies or profile pictures
            continue

        processed_images.append(imagePath)

        faces_list = []

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            faces_list.append([y, x+w, y+h, x])

        encodings = face_recognition.face_encodings(rgb, faces_list)
        for encoding in encodings:
            knownEncoding.append(encoding)
            knownNames.append(name)

    return processed_images


def process_images(image_classifier, images_dir, processed_images_directory):

    imagePaths = list(paths.list_images(images_dir))
    num_of_image = len(imagePaths)

    if num_of_image == 0:
        print("nothing to process")
        return None

    print("[INFO] processing images...")
    print(imagePaths)
    processed_images = register_faces(image_paths=imagePaths, classifier=image_classifier)
    move_processed_images(source_images=processed_images,
                          destination_folder_root=processed_images_directory)

    print("[INFO] images processed complete...")
	
    global knownEncoding
    global knownNames

    data = {"encoding": knownEncoding, "names": knownNames}

    print("[INFO] serialising encodings...")
    save_dataset(data)
    print("[INFO] serialising encodings done")


if __name__ == '__main__':

    face_dataset = os.path.join(
        current_app_path, "../data/registered_faces.pickle")
    cascPath = os.path.join(
        current_app_path, "../models/haarcascade_frontalface_default.xml")

    imagesFolder = os.path.join(current_app_path, "../data/faces/new")
    processedFolder = os.path.join(current_app_path, "../data/faces/processed")

    init_processed_directory(processedFolder)
    load_existing_dataset(face_dataset)
    process_images(image_classifier=cascPath, images_dir=imagesFolder,
                   processed_images_directory=processedFolder)
