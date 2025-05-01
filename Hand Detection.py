import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

# MediaPipe setup for hand gestures
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Global variables for adaptive settings
adaptive_brightness = 0  # Default brightness adjustment
adaptive_contrast = 1.0  # Default contrast multiplier
show_debug_windows = True  # Flag to enable/disable debug windows

# Resize factor for debug windows (smaller to fit on screen)
DEBUG_RESIZE_FACTOR = 0.5


# Function to count fingers in gesture mode
def count_fingers(hand_landmarks, hand_label):
    """
    Count the number of extended fingers on a hand

    This function checks each finger to see if it's extended by comparing
    the position of fingertips relative to their base (MCP) joints.
    The thumb is handled differently since it moves in a different axis.
    """
    tip_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
        mp_hands.HandLandmark.THUMB_TIP
    ]
    mcp_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP
    ]
    thumb_tip = hand_landmarks.landmark[tip_ids[4]]
    thumb_pip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    finger_count = 0

    # Thumb logic (different for left vs right hand)
    if hand_label == "Right":
        if thumb_tip.x < thumb_pip.x:
            finger_count += 1
    elif hand_label == "Left":
        if thumb_tip.x > thumb_pip.x:
            finger_count += 1

    # Check other fingers using MCP comparison for better accuracy
    for i in range(4):
        tip = hand_landmarks.landmark[tip_ids[i]]
        mcp = hand_landmarks.landmark[mcp_ids[i]]
        if tip.y < mcp.y:
            finger_count += 1
    return finger_count


def detect_gesture(hand_landmarks, hand_label):
    """
    Detect specific hand gestures based on finger positions

    Currently detects:
    - "Gun" gesture (all fingers extended)
    - "V" gesture (all fingers closed)
    """
    tip_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
        mp_hands.HandLandmark.THUMB_TIP
    ]
    mcp_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP
    ]
    fingers_open = []

    # Thumb check
    thumb_tip = hand_landmarks.landmark[tip_ids[4]]
    thumb_pip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    if hand_label == "Right":
        fingers_open.append(thumb_tip.x < thumb_pip.x)
    else:
        fingers_open.append(thumb_tip.x > thumb_pip.x)

    # Check other fingers using MCP comparison
    for i in range(4):
        tip = hand_landmarks.landmark[tip_ids[i]]
        mcp = hand_landmarks.landmark[mcp_ids[i]]
        fingers_open.append(tip.y < mcp.y)

    # Gesture recognition logic
    if all(fingers_open):  # All fingers open
        return "Gun"
    elif not any(fingers_open):  # All fingers closed
        return "V"
    return None


def resize_for_debug(image):
    """Resize image for debug windows to fit on screen"""
    h, w = image.shape[:2]
    new_h, new_w = int(h * DEBUG_RESIZE_FACTOR), int(w * DEBUG_RESIZE_FACTOR)
    return cv2.resize(image, (new_w, new_h))


def preprocess_image(frame):
    """
    Apply preprocessing filters to enhance hand detection in different lighting conditions

    Steps:
    1. Apply Gaussian blur to reduce noise
    2. Convert to YCrCb color space for better skin detection
    3. Apply adaptive histogram equalization for better contrast
    4. Apply brightness/contrast adjustments based on average brightness
    5. Apply skin color segmentation to focus on hand regions
    """
    global adaptive_brightness, adaptive_contrast

    # Keep the original for display
    original = frame.copy()

    # 1. Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # 2. Calculate average brightness and adjust parameters
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)

    # Dynamically adjust brightness/contrast based on scene
    if avg_brightness < 80:  # Low light conditions
        adaptive_brightness = min(adaptive_brightness + 1, 30)
        adaptive_contrast = min(adaptive_contrast + 0.02, 1.5)
    elif avg_brightness > 200:  # Very bright conditions
        adaptive_brightness = max(adaptive_brightness - 1, -10)
        adaptive_contrast = max(adaptive_contrast - 0.02, 0.8)
    else:  # Normal conditions - slowly return to defaults
        if adaptive_brightness > 0:
            adaptive_brightness -= 0.5
        elif adaptive_brightness < 0:
            adaptive_brightness += 0.5

        if adaptive_contrast > 1.0:
            adaptive_contrast -= 0.01
        elif adaptive_contrast < 1.0:
            adaptive_contrast += 0.01

    # 3. Apply adaptive brightness and contrast
    adjusted = cv2.convertScaleAbs(blurred, alpha=adaptive_contrast, beta=adaptive_brightness)

    # 4. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # 5. Apply skin color segmentation in YCrCb space (optional)
    ycrcb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YCrCb)

    # Show debug windows if enabled
    if show_debug_windows:
        # Add titles to each window
        cv2.putText(original, "1. Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(blurred, "2. Gaussian Blur", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(adjusted, f"3. Brightness/Contrast Adjusted", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(adjusted, f"Brightness: {adaptive_brightness:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(adjusted, f"Contrast: {adaptive_contrast:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(enhanced, "4. CLAHE Enhanced", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show each step in its own window
        cv2.imshow("1. Original Image", resize_for_debug(original))
        cv2.imshow("2. Gaussian Blur", resize_for_debug(blurred))
        cv2.imshow("3. Brightness/Contrast Adjusted", resize_for_debug(adjusted))
        cv2.imshow("4. CLAHE Enhanced", resize_for_debug(enhanced))

        # Add a visualization of the L channel before and after CLAHE
        l_before = cv2.cvtColor(l, cv2.COLOR_GRAY2BGR)
        cl_after = cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)
        cv2.putText(l_before, "5a. L Channel Before CLAHE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(cl_after, "5b. L Channel After CLAHE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("5a. L Channel Before CLAHE", resize_for_debug(l_before))
        cv2.imshow("5b. L Channel After CLAHE", resize_for_debug(cl_after))

    # Return the enhanced image
    return enhanced


# Function to update camera feed and process gestures
def update_camera():
    """
    Main camera loop that:
    1. Captures a frame from the camera
    2. Applies preprocessing to enhance the image
    3. Detects hand landmarks using MediaPipe
    4. Counts fingers and detects gestures
    5. Displays the results on screen
    """
    try:
        ret, frame = cap.read()
        if ret:
            # Apply preprocessing filters to enhance the frame
            enhanced_frame = preprocess_image(frame)

            # Convert the enhanced image to RGB for MediaPipe
            image_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # Use the enhanced frame for display
            display_image = enhanced_frame.copy()

            # Initialize direction text and finger counts
            left_finger_count = 0
            right_finger_count = 0

            # Process hand landmarks if detected
            hand_detected = False
            if results.multi_hand_landmarks:
                hand_detected = True
                for idx, (hand_landmarks, handedness) in enumerate(
                        zip(results.multi_hand_landmarks, results.multi_handedness)):

                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(display_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get handedness (left or right hand)
                    hand_label = handedness.classification[0].label

                    # Count fingers
                    finger_count = count_fingers(hand_landmarks, hand_label)

                    # Update finger count for the corresponding hand
                    if hand_label == "Left":
                        left_finger_count = finger_count
                    elif hand_label == "Right":
                        right_finger_count = finger_count

                    # Detect gesture
                    gesture = detect_gesture(hand_landmarks, hand_label)

                    # Display detected gesture
                    if gesture:
                        cv2.putText(display_image, f"Gesture: {gesture}", (10, 180),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Display finger counts on the image
            cv2.putText(display_image, f"Right Fingers: {right_finger_count}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_image, f"Left Fingers: {left_finger_count}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display adaptive settings for debugging
            cv2.putText(display_image, f"Brightness: {adaptive_brightness:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_image, f"Contrast: {adaptive_contrast:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Display hand detection status
            status = "Hand Detected" if hand_detected else "No Hand Detected"
            cv2.putText(display_image, status, (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Title for the final result window
            cv2.putText(display_image, "6. Final Result with Hand Detection", (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Show the final result in a separate window
            if show_debug_windows:
                cv2.imshow("6. Final Result with Hand Detection", resize_for_debug(display_image))

            # Update the camera display in the main Tkinter window
            img = Image.fromarray(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            camera_label.imgtk = imgtk
            camera_label.config(image=imgtk)

    except Exception as e:
        print(f"Error in camera update: {e}")

    # Continue updating camera feed
    camera_label.after(33, update_camera)  # ~30 FPS


# Function to toggle debug windows
def toggle_debug_windows():
    global show_debug_windows
    show_debug_windows = not show_debug_windows

    if not show_debug_windows:
        # Close all OpenCV windows when debug is disabled
        cv2.destroyAllWindows()

    # Update button text
    debug_button.config(text="Hide Debug Windows" if show_debug_windows else "Show Debug Windows")


# Main application setup
root = tk.Tk()
root.title("Enhanced Hand Gesture Recognition")
root.geometry("800x600")
root.configure(bg="#f0f0f0")

# Camera feed display
camera_label = tk.Label(root)
camera_label.pack(fill=tk.BOTH, expand=True, pady=5)

# Add control panel
control_frame = tk.Frame(root, bg="#d0d0d0")
control_frame.pack(fill=tk.X, pady=5)

# Label for app info
info_label = tk.Label(control_frame,
                      text="Hand Gesture Recognition - Adaptive to different lighting conditions",
                      font=("Arial", 12), bg="#d0d0d0")
info_label.pack(pady=5)

# Button to toggle debug windows
debug_button = tk.Button(control_frame, text="Hide Debug Windows", command=toggle_debug_windows)
debug_button.pack(pady=5)

# Initialize camera
cap = cv2.VideoCapture(0)  # Change to appropriate camera index if needed

# Start camera updates
update_camera()


# Clean up when closing
def on_closing():
    """
    Proper cleanup when closing the application
    """
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()  # Close all OpenCV windows
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the main loop
root.mainloop()