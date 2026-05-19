import cv2
import mediapipe as mp
import csv
import pickle
import os
import numpy as np
#Setup variables
HandLandmarker = mp.tasks.vision.HandLandmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
signs = ["yes", "no", "please", "thank_you"]

#Opening each video and reading each frame
options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path='hand_landmarker.task'), 
    running_mode = VisionRunningMode.IMAGE)
with HandLandmarker.create_from_options(options) as landmarker:
    for sign in signs:
        for filename in os.listdir(sign):
            print(filename)
            cap = cv2.VideoCapture(sign + "/" + filename)
            if not cap.isOpened():
                continue
            sequence = []
            while cap.isOpened():
                success, img = cap.read()
                if not success:
                    break
                #Convert for mediapipe
                rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                #Use the instance to DETECT landmarks
                result = landmarker.detect(mp_image)

                #Appending coordinates of each frame
                if result.hand_landmarks:
                    row = []
                    for lm in result.hand_landmarks[0]:
                        row.append(float(lm.x))
                        row.append(float(lm.y))
                        row.append(float(lm.z))
                    sequence.append(row)
            #Normalizing sequences
            if(len(sequence)<10):
                continue
            else:
                indices = np.linspace(0, len(sequence)-1,45, dtype = int )
                normalized_seq = []
                for i in indices:
                    normalized_seq.append(sequence[i])
                #storing each data per frame into csv files
                with open('data/' + sign + '/' + filename + '_data.csv', 'a') as f:
                    writer = csv.writer(f)
                    for n in normalized_seq:
                        writer.writerow(n)


