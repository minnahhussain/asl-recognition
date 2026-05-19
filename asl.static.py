import cv2
import mediapipe as mp
import csv
import pickle

#Setup variables
HandLandmarker = mp.tasks.vision.HandLandmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

#Open the camera (0 is default)
videoCap = cv2.VideoCapture(0)


#Create hand landmarker instance
#Setup options
options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path='hand_landmarker.task'), 
    running_mode = VisionRunningMode.IMAGE)


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

#Create the instance using with (opening and closing model)
with HandLandmarker.create_from_options(options) as landmarker:
    while videoCap.isOpened():
        success, img = videoCap.read()
        if not success:
            break

        #Flip the image
        img = cv2.flip(img,1)
        h,w,_ = img.shape

        #Convert for mediapipe
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        #Use the instance to DETECT landmarks
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            row = []
            if result.hand_landmarks:
                for lm in result.hand_landmarks[0]:
                    row.append(float(lm.x))
                    row.append(float(lm.y))
                    row.append(float(lm.z))
            prediction = model.predict([row])
            proba = model.predict_proba([row])
            print(prediction[0], proba)
            cv2.putText(img, prediction[0], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.imshow("ASL Collector", img)


        


        #whats clicked?
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # 27 is the Escape key
            break
        elif key != 255 and result.hand_landmarks:  # 255 means no key was pressed
            pressed_key = chr(key)  # converts the number back to a letter
         # now save your row with pressed_key as the label
            row=[pressed_key]
            for landmark in result.hand_landmarks[0]:
                row.append(landmark.x)
                row.append(landmark.y)
                row.append(landmark.z)
                
                
            with open('hand_data.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(row)

videoCap.release()





