import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

the_coords = deque(maxlen=2)
the_coords.append([0, 0, 0])
the_coords.append([0, 0, 0])

streaming = False
iteration_count = 0



def toggle_stream():
    global streaming
    if not streaming:
        streaming = True
        show_frame()
        toggle_button.config(text="Stop Stream")
    else:
        streaming = False
        toggle_button.config(text="Start Stream")

def angle_between_points(classe1, classe2, classe3):
    a = [classe1.x, classe1.y, classe1.z]
    b = [classe2.x, classe2.y, classe2.z]
    c = [classe3.x, classe3.y, classe3.z]

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ab = b - a
    bc = c - b

    dot_product = np.dot(ab, bc)
    ab_magnitude = np.linalg.norm(ab)
    bc_magnitude = np.linalg.norm(bc)

    angle_rad = np.arccos(dot_product / (ab_magnitude * bc_magnitude))
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def two_d_angle(classe1, classe2, classe3):
    a = [classe1.x, classe1.y]
    b = [classe2.x, classe2.y]
    c = [classe3.x, classe3.y]

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ab = b - a
    bc = c - b

    dot_product = np.dot(ab, bc)
    ab_magnitude = np.linalg.norm(ab)
    bc_magnitude = np.linalg.norm(bc)

    angle_rad = np.arccos(dot_product / (ab_magnitude * bc_magnitude))
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def show_frame():
    global streaming, iteration_count
    if streaming:
        _, frame = cap.read()
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(frame_rgb)
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            waist = pose_results.pose_landmarks.landmark[24]
            shoulder = pose_results.pose_landmarks.landmark[12]
            elbow = pose_results.pose_landmarks.landmark[14]
            wrist = pose_results.pose_landmarks.landmark[16]
            finger = pose_results.pose_landmarks.landmark[20]

            shoulder_angle = 180 - two_d_angle(elbow, shoulder, waist)
            elbow_angle = two_d_angle(shoulder, elbow, wrist)
            wrist_angle = two_d_angle(elbow, wrist, finger)

            arduino_shoulder_angle = shoulder_angle
            arduino_elbow_angle = elbow_angle
            arduino_wrist_angle = wrist_angle

            the_coords.append([arduino_shoulder_angle, arduino_elbow_angle, arduino_wrist_angle])

            if iteration_count % 30 == 0:
                the_change_in_angle = [the_coords[1][0] - the_coords[0][0], the_coords[1][1] - the_coords[0][1], the_coords[1][2] - the_coords[0][2]]
                print("\n\nTHE ANGLE IS AT SHOULDER:", shoulder_angle)
                print("THE ANGLE IS AT ELBOW:", elbow_angle)
                print("THE ANGLE IS AT WRIST:", wrist_angle)
                print("THE COORDS:", the_coords[0], the_coords[1])
                print("THE CHANGE IN ANGLE:", the_change_in_angle)

            info_label.config(text=f"Shoulder Angle: {int(shoulder_angle)}\nElbow Angle: {int(elbow_angle)}\nWrist Angle: {int(wrist_angle)}")

        except:
            pass
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        # Create an image with margins
        new_height = img.height + 2 * 30
        new_img = Image.new("RGB", (img.width, new_height), color="#add8e6")
        new_img.paste(img, (0, 30))  # Paste the original image with a top margin of 30 pixels

        img_tk = ImageTk.PhotoImage(image=new_img)
        label.imgtk = img_tk
        label.configure(image=img_tk)

        label.after(10, show_frame)  # Update every 10 milliseconds


        iteration_count += 1



root = tk.Tk()
root.title("Pose Estimation")
root.attributes('-fullscreen', True)  # Open in full screen
root.configure(bg="#add8e6")  # Set background color to light blue

# Add space at the top
# Add space at the top
tk.Label(root, text="", bg="#add8e6", height=1).pack()

# Header label with padding
header_label = tk.Label(root, text="AutoSurgeon", bg="#004080", fg="white", font=("Geist Sans", 45, "bold"))
header_label.pack(padx=50, pady=10)



# Add space at the bottom
tk.Label(root, text="", bg="#add8e6", height=1).pack()




style = ttk.Style()
style.configure('TButton', font=('Arial', 30), bg=('red'))


label = tk.Label(root)
label.pack()

info_label = tk.Label(root, text="", bg="#9fc5e8", fg="white", font=("Geist Sans", 15, "bold"))
info_label.pack()



# Add space at the top
tk.Label(root, text="", bg="#add8e6").pack()

# Create the button
toggle_button = tk.Button(root, text="Start Stream", command=toggle_stream, bg="#004080", fg="white", font=("Geist Sans", 25, "bold"), relief=tk.FLAT, bd=0)
toggle_button.pack()

# Add space at the bottom
tk.Label(root, text="", bg="#add8e6").pack()




root.mainloop()

cap.release()
cv2.destroyAllWindows()
