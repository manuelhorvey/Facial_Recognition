import face_recognition as fr 
import os
import cv2
import numpy as np 
import face_recognition 
from time import sleep
import datetime
import time


"""def get_user_data():
#import pymysql
    #Establish a connection a database
    conn = pymysql.connect(
        dbname="",
        user="",
        password="",
        host=""
    )
    #create a cursor connection
    cur = conn.cursor()

    #Execuute a query to fetch the usr data
    cur.execute("select name, image_path from users")

    #Fetch all the rows
    rows = cur.fetchall()

    return rows
"""
def get_encoded_faces():

    """
    looks through the faces folder and encodes all
    the faces

    :return: dict of (name, image encoded)
    """
    """    #fetch the user data
    users = get_user_data()

    for user in users:
        name, image_path =user
        face = fr.load_image_file(image_path)
        encoding = fr.face_encodings(face)[0]
        encoded[name] = encoding
    return encoded"""

    
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./facial_recognition/faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg"):
                face = fr.load_image_file(os.path.join(dirpath, f))
                face_encodings = fr.face_encodings(face)
                if len(face_encodings) > 0:
                    encoding = face_encodings[0]
                    encoded[f.split(".")[0]] = encoding

    return encoded

def unknown_image_encoded(img):
    """Encodes a file given the file name """
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
    # Create a VideoCapture object to capture images from the camera
    video_capture = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video capture.")
        return

    # Continuously display the live camera feed until the capture button is pressed
    while True:
        # Capture a frame from the camera
        ret, frame = video_capture.read()

        # Check if image capture was successful
        if not ret:
            print("Error: Could not capture image.")
            break

        # Display the captured frame
        cv2.imshow("", frame)

        # Wait for the user to press the capture button (spacebar)
        key = cv2.waitKey(1)
        if key == ord(' '):  # Spacebar key
            # Prompt the user to enter the desired image name
            image_name = input("Enter the name for the image (without extension): ")

            # Generate the image filename with the provided name
            image_filename = f"{image_name}.jpg"

            # Save the captured image to the images folder
            image_path = os.path.join(images_folder, image_filename)
            cv2.imwrite(image_path, frame)

            print("Image saved successfully:", image_path)
            break

    # Release the VideoCapture object and close any open windows
    video_capture.release()
    cv2.destroyAllWindows()

# Example usage:
images_folder = "./facial_recognition/faces"  # Path to the folder where images will be saved

