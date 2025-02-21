import cv2
import mediapipe as mp


def count_fingers(hand_landmarks):
    """
    Counts the number of open fingers (0-5) for a single hand.
    Args:
        hand_landmarks: MediaPipe hand landmarks for a single hand.
    Returns:
        int: Number of open fingers (0-5).
    """
    # Landmark indices for finger tips and middle joints
    tip_ids = [mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
               mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
               mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
               mp.solutions.hands.HandLandmark.PINKY_TIP,
               mp.solutions.hands.HandLandmark.THUMB_TIP]

    pip_ids = [mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP,
               mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP,
               mp.solutions.hands.HandLandmark.RING_FINGER_PIP,
               mp.solutions.hands.HandLandmark.PINKY_PIP,
               mp.solutions.hands.HandLandmark.THUMB_IP]

    # Count open fingers
    finger_count = 0

    # Check thumb separately cause horz
    thumb_tip = hand_landmarks.landmark[tip_ids[4]]
    thumb_pip = hand_landmarks.landmark[pip_ids[4]]
    #checking which hand so it counts it the right way
    if hand_label == "Right":
        if thumb_tip.x < thumb_pip.x:
            finger_count += 1
    elif hand_label == "Left":
        if thumb_tip.x > thumb_pip.x:
            finger_count += 1

    # Check other fingers
    for i in range(4):  # Index, Middle, Ring, Pinky
        tip = hand_landmarks.landmark[tip_ids[i]]
        pip = hand_landmarks.landmark[pip_ids[i]]
        if tip.y < pip.y:  # Finger is open
            finger_count += 1

    return finger_count


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# OpenCV video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Convert back to BGR for OpenCV display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Initialize direction text and finger counts
    direction = ""
    left_finger_count = 0
    right_finger_count = 0

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw hand landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get handedness (left or right hand)
            hand_label = handedness.classification[0].label

            # Count fingers
            finger_count = count_fingers(hand_landmarks)

            # Update finger count for the corresponding hand
            if hand_label == "Left":
                left_finger_count = finger_count
            elif hand_label == "Right":
                right_finger_count = finger_count

            # Get landmark coordinates
            landmarks = hand_landmarks.landmark
            wrist = landmarks[mp_hands.HandLandmark.WRIST]  # Wrist landmark
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]  # Index finger tip
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]  # Thumb tip
            middlefinger_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]  # Middle finger tip

            # Calculate relative position of index finger tip to wrist
            dx = index_tip.x - wrist.x
            dy = index_tip.y - wrist.y

            # Determine pointing direction based on dx and dy
            if abs(dx) > abs(dy):  # Horizontal movement dominates
                if dx > 0:
                    direction = "Right"
                else:
                    direction = "Left"
            else:  # Vertical movement dominates
                if dy > 0:
                    direction = "Backward"
                else:
                    direction = "Forward"

    # Display direction and finger counts on the image
    cv2.putText(image, f"Direction: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"Left Hand Fingers: {left_finger_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"Right Hand Fingers: {right_finger_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Hand Gesture', image)

    # Exit on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()