from TEST2 import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Define label mapping
label_map = {label: num for num, label in enumerate(actions)}

# Initialize lists to store sequences and labels
sequences, labels = [], []

# Define maximum sequence length and frame shape
max_sequence_length = 30
frame_shape = None

# Iterate through actions and sequences
for action in actions:
    for sequence in range(no_sequences):
        window = []
        consistent_shape = True  # Flag to track if all frames within the sequence have the same shape
        for frame_num in range(sequence_length):
            try:
                # Load frame data
                frame = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)
                
                # Record the shape of the first frame
                if frame_shape is None:
                    frame_shape = frame.shape
                elif frame.shape != frame_shape:
                    consistent_shape = False
                    break  # Exit the loop if inconsistent shape is found
                
                # Append frame to the window
                window.append(frame)
            except Exception as e:
                print(f"Error loading file: {e}")
                continue  # Skip this frame and continue with the next one
        
        # Check if all frames within the sequence have the same shape and if the sequence length is valid
        if consistent_shape and len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])

# Convert sequences to numpy array
X = np.array(sequences)

# Check the shape of sequences
print("Shape of X:", X.shape)

# Convert labels to categorical
y = to_categorical(labels).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Define the model architecture
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length,) + frame_shape))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save the model

model.save('mymodel.h5')
