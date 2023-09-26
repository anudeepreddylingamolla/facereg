import cv2
import numpy as np
import mtcnn
from architecture import *
from train_v2 import normalize, l2_normalizer
from scipy.spatial.distance import cosine
import pickle
import pandas as pd
import os

confidence_t = 0.99
recognition_t = 0.5
required_size = (160, 160)

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def generate_unique_id(existing_ids):
    new_id = max(existing_ids) + 1 if existing_ids else 1
    return new_id

def detect(img, detector, encoder, encoding_dict, recognized_names):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)

    recognized_ids = {}

    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        if name != 'unknown':
            if name not in recognized_ids:
                unique_id = generate_unique_id(list(recognized_ids.values()))
                recognized_ids[name] = unique_id
                recognized_names.add(name)  # Add the unique name to the set

    return recognized_names  # Return the recognized names

if __name__ == "__main__":
    required_shape = (160, 160)
    face_encoder = InceptionResNetV2()
    path_m = "facenet_keras_weights.h5"
    face_encoder.load_weights(path_m)
    encodings_path = 'encodings/encodings.pkl'
    face_detector = mtcnn.MTCNN()
    encoding_dict = load_pickle(encodings_path)

    folder_path = "outputofdetection"  # Change this to the path of your input folder containing images
    recognized_names = set()  # Use a set to store unique recognized names

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            frame = cv2.imread(image_path)
            names = detect(frame, face_detector, face_encoder, encoding_dict, recognized_names)

    # Save recognized names to an Excel file
    excel_filename = "recognized_faces.xlsx"
    df = pd.DataFrame({"Name": list(names)})
    df.to_excel(excel_filename, index=False)

    print(f"Recognized names saved to {excel_filename}")
