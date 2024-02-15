import face_recognition as fr 
import os
import cv2
import numpy as np 
import face_recognition 
from time import sleep
import datetime


def get_encoded_faces():
    encoded = {} 

    for dirpath, dnames, fnames in os.walk("./facial_recognition/faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg"):
                face = fr.load_image_file(os.path.join(dirpath, f))  
                face_encodings = fr.face_encodings(face)
                if face_encodings:  # Check if any faces were found
                    encoding = face_encodings[0]  # Take the first face encoding
                    encoded[f.split(".")[0]] = encoding  
    return encoded


def unknown_image_encoded(img):
    face = fr.load_image_file("/facial_recognition/faces/" + img)
    encoding = fr.face_encodings(face)[0]  
    return encoding


def classify_face_live(should_continue):
    print("Starting face recognition...")
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open video capture.")
        return

    recognized_names = set()

    while should_continue.is_set():
        ret, frame = video_capture.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        face_locations = face_recognition.face_locations(frame)
        unknown_face_encodings = face_recognition.face_encodings(frame, face_locations)

        face_names = []
        for face_encoding in unknown_face_encodings:
            matches = face_recognition.compare_faces(faces_encoded, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

            if name not in recognized_names:
                with open('attendance.txt', 'a') as f:
                    f.write(f'{name}, {datetime.datetime.now()}\n')
                recognized_names.add(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)
            cv2.rectangle(frame, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print("Face recognition stopped.")


def capture_and_save_image(images_folder):
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Error: Could not capture image.")
            break

        cv2.imshow("Register a User", frame)

        key = cv2.waitKey(1)
        if key == ord(' '):
            break

    image_filename = "user_image.jpg"
    image_path = os.path.join(images_folder, image_filename)
    cv2.imwrite(image_path, frame)

    print("Image saved successfully:", image_path)

    video_capture.release()
    cv2.destroyAllWindows()


def main():
    should_continue = threading.Event()
    should_continue.set()
    classify_face_live(should_continue)

if __name__ == "__main__":
    import threading
    main()
