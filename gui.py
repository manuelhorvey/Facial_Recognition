import tkinter as tk
from PIL import Image, ImageTk
import cv2
import threading
import face_recognition as fr 
import os
import numpy as np 
from time import sleep
import datetime
from face_rec import classify_face_live, capture_and_save_image

# Create an Event object
should_continue = threading.Event()

def start_session():
    global should_continue
    global video_capture
    # Set the event to signal the worker thread to continue running
    should_continue.set()
    video_capture = cv2.VideoCapture(0)
    threading.Thread(target=classify_face_live, args=(should_continue,)).start()
    

def stop_session():
    global should_continue
    # Clear the event to signal the worker thread to stop
    should_continue.clear()

def register_user(images_folder):
    # Capture and save an image of the user
    capture_and_save_image(images_folder)

# Create the main window
root = tk.Tk()
root.title("Face Recognition System")
root.protocol("WM_DELETE_WINDOW")

# Create a canvas for the video frame
canvas = tk.Canvas(root, width=700, height=500)
canvas.pack()

# Create a label for instructions
label = tk.Label(root, text="Press 'Start Session' to begin recognition", font=("Arial", 12))
label.pack(pady=10)

# Create a frame to group buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=5)

# Create a 'Start a Session' button
start_button = tk.Button(button_frame, text="Start Session", command=start_session, width=20)
start_button.grid(row=0, column=0, padx=5)

# Create a 'Stop' button
stop_button = tk.Button(button_frame, text="Stop", command=stop_session, width=20)
stop_button.grid(row=0, column=1, padx=5)

# Create a 'Register a User' button
register_button = tk.Button(root, text="Register User", command=lambda: register_user("./facial_recognition/faces/"), width=20)
register_button.pack(pady=10)

# Run the main loop
root.mainloop()
