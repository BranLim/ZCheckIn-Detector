# face_registration.py
from imutils import paths
import face_recognition
import pickle
import cv2
import os


def register_face(image_path, detected_faces, face_name_mapping):
    assert isinstance(image_path, str)
    assert isinstance(detected_faces, list)
    assert isinstance(face_name_mapping, dict)

    name = image_path.split(os.path.sep)[-2]
    image = cv2.imread(image_path)

    print(name)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    current_encodings = face_name_mapping.get("encoding")
    names = face_name_mapping.get("names")

    if current_encodings is None:
        current_encodings = []
        face_name_mapping["encoding"] = current_encodings

    if names is None:
        names = []
        face_name_mapping["names"] = names

    
    encodings = face_recognition.face_encodings(rgb_image, detected_faces)
    for encoding in encodings:
        current_encodings.append(encoding)
        names.append(name)

    return face_name_mapping


def save_registered_faces(face_data_path, face_name_mappings):
    assert isinstance(face_data_path, str)
    assert isinstance(face_name_mappings, dict)

    if face_name_mappings:
        with open(face_data_path, 'wb') as f:
            f.write(pickle.dumps(face_name_mappings))
