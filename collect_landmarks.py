import cv2  #OpenCV - used to read images from disk
import mediapipe as mp #MediaPipe - used to detect hand landmarks
import numpy as np
import pandas as pd
import os

mp_hands = mp.solutions.hands #Loads MediaPipe's hand detection module
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

DATA_DIR = "data1"
OUTPUT_DIR = "data1/landmarks" #Where the output CSV will be saved
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = ["rock", "paper", "scissors"]

#Function that takes an image path and returns landmarks
def extract_landmarks(image_path):
    image = cv2.imread(image_path) #Reads the image from disk as a NumPy array
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Converts from BGR to RGB - OpenCV loads images in BGR by default but MediaPipe expects RGB
    result = hands.process(image_rgb) #Runs MediaPipe hand detection on the image, returns detected landmarks
    
    #Checks if any hand was detected
    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0] #Takes the first detected hand's landmarks
        #Loops through all 21 landmarks, each has x, y, z coordinates -extends the list with all 3 values. Final result is 63 numbers per image
        coords = []
        for lm in landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        return coords
    return None

all_data = [] #Empty list to collect all extracted landmark rows

for label, cls in enumerate(CLASSES):
    folder = os.path.join(DATA_DIR, cls) #Constructs the path to the folder containing images for the current class (e.g., "data1/rock")
    images = os.listdir(folder) #Lists all files in the folder, which should be the images to process
    print(f"Processing {cls} — {len(images)} images")
    
    for img_file in images:
        img_path = os.path.join(folder, img_file)
        landmarks = extract_landmarks(img_path)
        
        #If landmarks were found, append the 63 coordinates plus the label as one r
        if landmarks:
            all_data.append(landmarks + [label])
        
print(f"\nTotal samples extracted: {len(all_data)}")

columns = [f"x{i}" for i in range(63)] + ["label"]
df = pd.DataFrame(all_data, columns=columns)
df.to_csv(os.path.join(OUTPUT_DIR, "landmarks.csv"), index=False)
print(f"Saved to {OUTPUT_DIR}/landmarks.csv")