import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('model/emotion_model.hdf5')

# Emotion labels
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

# Emotion smoothing
emotion_history = []

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Camera not working")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        prediction = model.predict(face, verbose=0)
        emotion = labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Smooth predictions
        emotion_history.append(emotion)
        emotion_history = emotion_history[-10:]
        emotion = max(set(emotion_history), key=emotion_history.count)

        text = f"{emotion} ({confidence:.2f})"

        # Draw box
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)

    cv2.imshow('Face Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()