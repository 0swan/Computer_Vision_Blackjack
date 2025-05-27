import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# --- Configuration ---
# Replace 'your_model.h5' with the actual path to your Keras .h5 model file.
MODEL_PATH = 'keras.model.h5'
# Define the target image size that your model expects.
# For example, if your model was trained on 224x224 RGB images, use (224, 224).
TARGET_IMAGE_SIZE = (224, 224)
# Define the class labels that your model predicts.
# Replace with your actual class names in the correct order.
CLASS_LABELS = ['A', '2', '3', '4', '5','6','7','8','9','face'] # Example labels

def load_keras_model(model_path):
    """
    Loads a Keras model from an .h5 file.
    """
    try:
        model = keras.models.load_model(model_path)
        print(f"Model '{model_path}' loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model path is correct and the file is a valid Keras .h5 model.")
        return None

def preprocess_frame(frame, target_size):
    """
    Preprocesses a single video frame for model prediction.
    This typically involves resizing and normalizing pixel values.
    Adjust preprocessing steps based on how your model was trained.
    """
    # Resize the image to the target size expected by the model
    resized_frame = cv2.resize(frame, target_size)

    # Convert BGR to RGB (OpenCV reads images as BGR, Keras models usually expect RGB)
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Normalize pixel values. Common normalization methods:
    # 1. Scale to [0, 1]: Divide by 255.0
    # 2. Scale to [-1, 1]: (pixel / 127.5) - 1
    # Choose the one that matches your model's training.
    normalized_frame = rgb_frame / 255.0  # Example: scale to [0, 1]

    # Expand dimensions to create a batch of 1 image (e.g., (1, height, width, channels))
    preprocessed_frame = np.expand_dims(normalized_frame, axis=0)

    return preprocessed_frame

def main():
    """
    Main function to capture webcam feed, preprocess frames,
    and make predictions using the loaded Keras model.
    """
    # Load the Keras model
    model = load_keras_model(MODEL_PATH)
    if model is None:
        return

    # Initialize the webcam (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam. Please check if the webcam is connected and not in use.")
        return

    print("\nPress 'q' to quit the webcam feed.")
    print("Capturing video stream...")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame, exiting...")
            break

        # Preprocess the frame for the model
        processed_frame = preprocess_frame(frame, TARGET_IMAGE_SIZE)

        # Make a prediction
        try:
            predictions = model.predict(processed_frame)
            # Get the predicted class index (the one with the highest probability)
            predicted_class_index = np.argmax(predictions[0])
            # Get the probability of the predicted class
            confidence = predictions[0][predicted_class_index]
            predicted_label = CLASS_LABELS[predicted_class_index]

            # Display the prediction on the frame
            text = f"Prediction: {predicted_label} ({confidence:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error during prediction: {e}")
            cv2.putText(frame, "Prediction Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the original frame with the prediction
        cv2.imshow('Webcam Feed - Keras Prediction', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam feed stopped.")

if __name__ == "__main__":
    main()



def get_card_value(card):
    """Converts face cards to numbers, Ace stays as 'A'."""
    if card in ['Face']:
        return 10
    elif card == 'A':
        return 'A'
    else:
        return int(card)

def is_pair(hand):
    return len(hand) == 2 and hand[0] == hand[1]

def is_soft(hand):
    return 'A' in hand and sum(get_card_value(c) if c != 'A' else 1 for c in hand) <= 10

def hand_total(hand):
    values = [get_card_value(c) for c in hand]
    total = sum(v if v != 'A' else 11 for v in values)
    if total > 21 and 'A' in hand:
        total -= 10
    return total

def blackjack_action(player_hand, dealer_card):
    dealer_val = get_card_value(dealer_card)
    total = hand_total(player_hand)

    # Handle splitting pairs
    if is_pair(player_hand):
        card = player_hand[0]
        if card in ['A', '8']:
            return 'Split'
        elif card in ['10', 'Face']:
            return 'Stand'
        elif card == '9':
            return 'Split' if dealer_val not in [7, 10, 'A'] else 'Stand'
        elif card == '7':
            return 'Split' if dealer_val <= 7 else 'Hit'
        elif card == '6':
            return 'Split' if dealer_val <= 6 else 'Hit'
        elif card == '4':
            return 'Split' if dealer_val in [5, 6] else 'Hit'
        elif card == '3' or card == '2':
            return 'Split' if dealer_val <= 7 else 'Hit'

    # Handle soft hands (contain an Ace)
    if is_soft(player_hand):
        if total >= 19:
            return 'Stand'
        elif total == 18:
            return 'Stand' if dealer_val in [2, 7, 8] else 'Hit'
        else:
            return 'Hit'

    # Handle hard hands (no usable Ace)
    if total >= 17:
        return 'Stand'
    elif 13 <= total <= 16:
        return 'Stand' if dealer_val <= 6 else 'Hit'
    elif total == 12:
        return 'Stand' if 4 <= dealer_val <= 6 else 'Hit'
    else:
        return 'Hit'
