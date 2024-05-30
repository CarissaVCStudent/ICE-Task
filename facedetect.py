import cv2
import numpy as np
import os
import npwriter

name = "Nicholas Cage"
image_folder = r'C:\Users\carissa\Desktop\NicholasCage'
f_list = []

# Provide the full path to the Haar Cascade XML file
cascade_path = r'C:\Users\carissa\Desktop\ICE\haarcascade_frontalface_default'
classifier = cv2.CascadeClassifier(cascade_path)

# Check if the classifier is loaded correctly
if classifier.empty():
    print(f"Error loading cascade classifier from {cascade_path}")
else:
    print(f"Successfully loaded cascade classifier from {cascade_path}")

for filename in os.listdir(image_folder):
    img_path = os.path.join(image_folder, filename)
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Error reading image {img_path}")
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.5, 5)
    
    if len(faces) == 0:
        print(f"No faces found in {img_path}")
    else:
        print(f"Faces found in {img_path}: {len(faces)}")

    for (x, y, w, h) in faces:
        im_face = frame[y:y + h, x:x + w]
        gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (100, 100))
        f_list.append(gray_face.reshape(-1))
        if len(f_list) == 10:
            break
    if len(f_list) == 10:
        break

if len(f_list) == 0:
    print("No faces detected to write to CSV")
else:
    npwriter.write(name, np.array(f_list))
    print("Face data written to CSV")
