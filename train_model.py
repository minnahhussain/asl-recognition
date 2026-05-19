import os
import csv
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
x = []
y =[]

#scaling for diff hand sizes
def augment_scale(sequence):
    new_seq = []
    for row in sequence:
        scaled_row = []
        for val in row:
            scaled_row.append(val*np.random.uniform(0.8,1.2))
        new_seq.append(scaled_row)
    return new_seq
#adjusting between r and l hands
def augment_mirror(sequence):
    new_seq=[]
    for row in sequence:
        scaled_row=[]
        for i, val in enumerate(row):
            if(i%3==0):
                scaled_row.append(1-val)
            else:
                scaled_row.append(val)
        new_seq.append(scaled_row)
    return new_seq
#adjusting for noise
def augment_noise(sequence):
    new_seq = []
    for row in sequence:
        scaled_row = []
        for val in row:
            scaled_row.append(val+np.random.uniform(-0.02,0.02))
        new_seq.append(scaled_row)
    return new_seq

for sign in os.listdir('data/'):
    if sign.startswith('.'):
        continue
    for filename in os.listdir('data/' + sign + '/'):
        with open('data/' + sign + '/' + filename, 'r') as f:
            reader = csv.reader(f)
            rows = [[float(val) for val in row] for row in reader]
            x.append(rows)
            y.append(sign)
for sequence, label in zip(x.copy(), y.copy()):
    x.append(augment_scale(sequence))
    y.append(label)
    x.append(augment_mirror(sequence))
    y.append(label)
    x.append(augment_noise(sequence))
    y.append(label)
#convert to numpy arrays
x = np.array(x, dtype=float)
y = np.array(y)

#label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
#split into training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#creates memory, and finds patterns between values
model = Sequential([
    LSTM(64, input_shape=(45, 63), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])
#measures accuracy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#trains model
model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

#saves model
model.save('asl_lstm_model.keras')

