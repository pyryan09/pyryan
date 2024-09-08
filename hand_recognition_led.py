import cv2
import mediapipe as mp
import serial
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize serial communication with Arduino
arduino = serial.Serial('COM3', 9600)  # Update 'COM3' with your port name
time.sleep(2)  # Wait for the serial connection to initialize

# Function to count extended fingers
def count_fingers(hand_landmarks):
    # Thumb tip and base landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_base = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]

    # Check if thumb is extended (using y-coordinate as a heuristic)
    thumb_extended = thumb_tip.y < thumb_base.y

    # Finger tips and base landmarks
    fingers = [thumb_extended]  # Initialize with thumb state

    for finger_tip_landmark, finger_base_landmark in zip(
        [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
         mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP],
        [mp_hands.HandLandmark.INDEX_FINGER_DIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP, 
         mp_hands.HandLandmark.RING_FINGER_DIP, mp_hands.HandLandmark.PINKY_DIP]
    ):
        finger_tip = hand_landmarks.landmark[finger_tip_landmark]
        finger_base = hand_landmarks.landmark[finger_base_landmark]

        # Check if finger is extended (using y-coordinate as a heuristic)
        extended = finger_tip.y < finger_base.y
        fingers.append(extended)

    return fingers.count(True)  # Count how many fingers are extended

# Open video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Count fingers
            num_fingers = count_fingers(hand_landmarks)
            
            # Send number of fingers detected to Arduino (1-5)
            if num_fingers == 0:
                arduino.write(b'0')  # Command to turn off all LEDs
            elif 1 <= num_fingers <= 5:
                arduino.write(str(num_fingers).encode())  # Send number of fingers to Arduino
    else:
        # No hands detected
        arduino.write(b'0')  # Command to turn off all LEDs

    # Draw hand landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Hand Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()