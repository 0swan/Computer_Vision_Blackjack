import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained Keras model
model = load_model('path_to_your_model.h5')

# Label list in the same order as your model's output
labels = ['ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'face']

# Map labels to Blackjack values
blackjack_values = {
    'ace': 11,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    '10': 10,
    'face': 10
}

def preprocess_frame(frame, target_size=(224, 224)):
    """Resize and normalize frame for prediction."""
    img = cv2.resize(frame, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Start webcam capture
cap = cv2.VideoCapture(0)
print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optionally crop the center of the frame if needed
    h, w, _ = frame.shape
    center_crop = frame[h//4:h*3//4, w//4:w*3//4]

    # Preprocess and predict
    processed = preprocess_frame(center_crop)
    prediction = model.predict(processed)
    class_index = np.argmax(prediction[0])
    label = labels[class_index]
    value = blackjack_values[label]

    # Display result
    cv2.putText(frame, f"{label.upper()} (Value: {value})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.rectangle(frame, (w//4, h//4), (w*3//4, h*3//4), (255, 0, 0), 2)
    cv2.imshow("Card Detector", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
