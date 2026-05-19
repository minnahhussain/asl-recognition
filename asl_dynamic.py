import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import numpy as np

#Setup variables
HandLandmarker = mp.tasks.vision.HandLandmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

#Setup options
options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path='hand_landmarker.task'), 
    running_mode = VisionRunningMode.IMAGE)
#model 
model = load_model('asl_lstm_model.keras')
le = LabelEncoder()
le.fit(['no', 'please', 'thank_you', 'yes'])

#Open the camera (0 is default)
videoCap = cv2.VideoCapture(0)

#Create the instance using with (opening and closing model)
with HandLandmarker.create_from_options(options) as landmarker:
    sequence =[]
    current_prediction=""
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

         #Appending coordinates of each frame
        if result.hand_landmarks:
            row = []
            for lm in result.hand_landmarks[0]:
                row.append(float(lm.x))
                row.append(float(lm.y))
                row.append(float(lm.z))
            sequence.append(row)
        else:
            #creating prediction
            if(len(sequence)>=10):
                indices = np.linspace(0, len(sequence)-1, 45, dtype=int)
                normalized_seq = [sequence[i] for i in indices]
                input_data = np.array(normalized_seq).reshape(1, 45, 63)                
                prediction = model.predict(input_data)
                current_prediction = str(le.inverse_transform([np.argmax(prediction)]))
                for sign, prob in zip(le.classes_, prediction[0]):
                    print(f"{sign}: {prob:.2f}")
            sequence=[] #reset after predicting
        cv2.putText(img, current_prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.imshow("ASL Collector", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break
