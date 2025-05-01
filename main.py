import cv2
import mediapipe as mp
import speech_recognition as sr
import google.generativeai as genai
import json
import re
import tkinter as tk
from tkinter import scrolledtext, ttk, Frame, Label, Button, Entry
from PIL import Image, ImageTk
import threading
import time
import math
import serial.tools.list_ports
from pydobot import Dobot
import queue
import numpy as np

#Setting up Google Gemini API Key
genai.configure(api_key="AIzaSyD9aXNC1TtWMmu4a0iD77Dtmge89WPke-0")  # connecting this the app to the cloud LLM

#Global variables
dobot = None
connected = False
home_x, home_y, home_z = 200, 0, 20  # Default home position that we can adjust anytime for the dobot
drawing_z = -59  # Z position for drawing on the surface because the Dobot's base is higher than the table with aprox 6cm
mode = "normal"  #Current mode: "normal" or "gesture" that changes by changing the mode by pressing the button
stop_gesture_thread = threading.Event()
message_queue = queue.Queue()  # For thread-safe messaging

#MediaPipe setup for hand gestures detection later on
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

#Movement parameters for gesture control
#Step size for movement (reduced for more precision) and can be increased for a faster response
step_size_y = 30
step_size_z = 40
#to avoide the robot being jammed
workspace_limits = {
    "x_min": 150, "x_max": 250,
    "y_min": -150, "y_max": 150,
    "z_min": -300, "z_max": 100
}


def analyze_hand_gesture(hand_landmarks, hand_label):
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

    # Thumb check (different logic for left/right hand)
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

    # Count total fingers
    finger_count = sum(fingers_open)

    # Determine gesture
    gesture = None
    if all(fingers_open):  # All fingers open used for dropping the object
        gesture = "Gun"
    elif not any(fingers_open):  # All fingers closed
        gesture = "V"
    #for future gestures just add elif(s) here

    return finger_count, gesture

# Function to activate gesture control mode
def activate_gesture_mode():
    global mode, stop_gesture_thread

    if not connected:
        update_chat("Mike: Please connect the Dobot first before activating gesture mode.", "error")
        return

    # Reset the stop event
    stop_gesture_thread.clear()

    # Update UI elements
    mode = "gesture"
    update_chat("Mike: Hand gesture control mode activated! 👋 Make gestures to control the Dobot.", "mike")
    update_chat("Mike: V gesture = Suction On, Gun gesture = Suction Off, Point = Move", "system")
    update_chat("Mike: Say 'exit gesture mode' or type 'quit' to return to normal mode", "system")

    # Change button text
    voice_button.config(text="🎤 Exit Gesture Mode")

    # Start gesture processing thread
    threading.Thread(target=process_gestures, daemon=True).start()


# Global variables for adaptive processing with standard intial values before calculation
adaptive_brightness = 0
adaptive_contrast = 1.0


# Function to process gestures in a separate thread to make the flow smoother
def process_gestures():
    global dobot, mode, stop_gesture_thread, adaptive_brightness, adaptive_contrast

    last_movement_time = time.time()
    movement_cooldown = 0.5  # seconds between movements adjustable later on to balance between precision and fastre response

    while not stop_gesture_thread.is_set():
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            #Apply preprocessing to enhance hand detection (to make sure it is detected in hard situations)
            display_image = preprocess_image(frame)

            #Convert the preprocessed image to RGB for MediaPipe because openCV uses BGR and Mediapipe uses RGB
            image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            #Convert back to BGR for OpenCV display
            display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            direction = ""
            left_finger_count = 0
            right_finger_count = 0
            action = ""

            # Process hand landmarks if detected "the following part is mostly taken from Mediapipe library website)
            if results.multi_hand_landmarks:
                for idx, (hand_landmarks, handedness) in enumerate(
                        zip(results.multi_hand_landmarks, results.multi_handedness)):
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(display_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get handedness (left or right hand)
                    hand_label = handedness.classification[0].label

                    # Count fingers and return the gesture detected
                    finger_count, gesture = analyze_hand_gesture(hand_landmarks, hand_label)


                    # Update finger count for the corresponding hand
                    if hand_label == "Left":
                        left_finger_count = finger_count
                    elif hand_label == "Right":
                        right_finger_count = finger_count


                    #Check if enough time has passed since last movement adjusted in the begining of the function
                    current_time = time.time()
                    if current_time - last_movement_time < movement_cooldown:
                        continue

                    # Control suction and stop movement based on gesture
                    if gesture == "V":
                        if check_dobot_connection():
                            dobot.suck(True)  # Activate suction
                            action = "Suction On"
                            message_queue.put(("system", "Mike: Suction activated"))
                            last_movement_time = current_time
                            continue  # Skip further processing
                    elif gesture == "Gun":
                        if check_dobot_connection():
                            dobot.suck(False)  # Deactivate suction
                            action = "Suction Off"
                            message_queue.put(("system", "Mike: Suction deactivated"))
                            last_movement_time = current_time
                            continue  # Skip further processing


                    # Get landmark coordinates
                    landmarks = hand_landmarks.landmark
                    wrist = landmarks[mp_hands.HandLandmark.WRIST]  # Wrist landmark
                    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]  # Index finger tip

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
                            direction = "Down"
                        else:
                            direction = "Up"

                    # Move Dobot based on direction if connected
                    if check_dobot_connection():
                        current_position = dobot.pose()  # Get current position
                        x, y, z, r = current_position[:4]  # Extract only the first four values

                        if direction == "Up":
                            z = min(z + step_size_z, workspace_limits["z_max"])
                            message_queue.put(("system", f"Moving Up: Z = {z:.1f}"))
                        elif direction == "Down":
                            z = max(z - step_size_z, workspace_limits["z_min"])
                            message_queue.put(("system", f"Moving Down: Z = {z:.1f}"))
                        elif direction == "Right":
                            y = min(y + step_size_y, workspace_limits["y_max"])
                            message_queue.put(("system", f"Moving Right: Y = {y:.1f}"))
                        elif direction == "Left":
                            y = max(y - step_size_y, workspace_limits["y_min"])
                            message_queue.put(("system", f"Moving Left: Y = {y:.1f}"))

                        # Move the Dobot to the new position
                        dobot.move_to(x, y, z, r, wait=True)
                        last_movement_time = current_time

            # Display direction, finger counts, and action on the image
            cv2.putText(display_image, f"Mode: Gesture Control", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                        2)
            cv2.putText(display_image, f"Direction: {direction}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                        2)
            cv2.putText(display_image, f"L Fingers: {left_finger_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
            cv2.putText(display_image, f"R Fingers: {right_finger_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
            cv2.putText(display_image, f"Action: {action}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Update the camera display
            img = Image.fromarray(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            camera_label.imgtk = imgtk
            camera_label.config(image=imgtk)

            # Process any messages in the queue to display them in the chat box
            process_message_queue()

            # Brief delay to reduce CPU usage just for safety
            time.sleep(0.05)

            # Check if we need to exit the mode
            if mode != "gesture":
                break

        except Exception as e:
            message_queue.put(("error", f"Gesture processing error: {str(e)}"))
            time.sleep(1)  # Wait before retry

    # Restore normal camera view when exiting
    update_camera()


def preprocess_image(frame):
    """
    Applying preprocessing filters to enhance hand detection in different lighting conditions
    this is influenced by the Computer vision module by Professor Adrian F Clark (Uni of ESSEX)
    Steps:
    1. Apply Gaussian blur to reduce noise
    2. Convert to YCrCb color space for better skin detection
    3. Apply adaptive histogram equalization for better contrast
    4. Apply brightness/contrast adjustments based on average brightness
    5. Apply skin color segmentation to focus on hand regions
    """
    global adaptive_brightness, adaptive_contrast

    #Keep the original for display
    original = frame.copy()

    #Apply Gaussian blur to reduce noise and the ksize can be adjusted to increase or decrase the blur
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    #Calculate average brightness and adjust parameters using the gray scal histogram
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)

    #Dynamically adjust brightness/contrast based on scene
    if avg_brightness < 80:  # Low light conditions
        adaptive_brightness = min(adaptive_brightness + 1, 30)
        adaptive_contrast = min(adaptive_contrast + 0.02, 1.5)
    elif avg_brightness > 200:  # Very bright conditions
        adaptive_brightness = max(adaptive_brightness - 1, -10)
        adaptive_contrast = max(adaptive_contrast - 0.02, 0.8)
    else:  #Normal conditions - slowly return to defaults
        if adaptive_brightness > 0:
            adaptive_brightness -= 0.5
        elif adaptive_brightness < 0:
            adaptive_brightness += 0.5

        if adaptive_contrast > 1.0:
            adaptive_contrast -= 0.01
        elif adaptive_contrast < 1.0:
            adaptive_contrast += 0.01

    #Apply adaptive brightness and contrast
    adjusted = cv2.convertScaleAbs(blurred, alpha=adaptive_contrast, beta=adaptive_brightness)

    #Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    #Apply skin color segmentation in YCrCb space (optional)
    ycrcb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YCrCb)

    #Return enhanced image for processing through mediapipe
    return enhanced

# Function to deactivate gesture mode
def deactivate_gesture_mode():
    global mode, stop_gesture_thread

    # Set the stop event to terminate the gesture thread
    stop_gesture_thread.set()

    # Update mode and UI
    mode = "normal"
    voice_button.config(text="🎤 Speak")

    update_chat("Mike: Hand gesture control mode deactivated. Back to normal operation! 🤖", "mike")


# Function to process messages from the queue
def process_message_queue():
    try:
        while not message_queue.empty():
            message_type, message = message_queue.get_nowait()
            update_chat(message, message_type)
    except:
        pass


# Function to connect to Dobot
def connect_dobot():
    global dobot, connected
    try:
        # Close any existing connection first
        if dobot is not None:
            try:
                dobot.close()
                time.sleep(1)  # Give it time to properly disconnect
            except:
                pass
            dobot = None

        # Find available ports
        available_ports = serial.tools.list_ports.comports()
        port = None

        # Show connecting status
        update_chat("Mike: Searching for Dobot...", "system")
        connect_button.config(text="Connecting...", state=tk.DISABLED)
        root.update()

        for p in available_ports:
            port = p.device
            update_chat(f"Mike: Found port: {port}", "system")
            break

        if port:
            #timeout and retry mechanism
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    dobot = Dobot(port=port, verbose=False)
                    # Test if connection works by requesting position
                    (x, y, z, r, j1, j2, j3, j4) = dobot.pose()
                    update_chat(f"Mike: Connected to Dobot Magician! Current position: X={x:.2f}, Y={y:.2f}, Z={z:.2f}",
                                "system")
                    connect_button.config(text="Connected ✓", state=tk.DISABLED)
                    reconnect_button.config(state=tk.NORMAL)
                    home_button.config(state=tk.NORMAL)
                    connected = True
                    return True
                except Exception as e:
                    update_chat(f"Mike: Connection attempt {attempt + 1}/{max_attempts} failed: {str(e)}", "system")
                    if dobot is not None:
                        try:
                            dobot.close()
                        except:
                            pass
                        dobot = None
                    time.sleep(2)  # Wait before retry

            update_chat("Mike: Failed to connect after multiple attempts.", "error")
            connect_button.config(text="Connect Dobot", state=tk.NORMAL)
            return False
        else:
            update_chat("Mike: No Dobot found. Please connect your Dobot and retry.", "error")
            connect_button.config(text="Connect Dobot", state=tk.NORMAL)
            return False
    except Exception as e:
        update_chat(f"Mike: Failed to connect to Dobot: {str(e)}", "error")
        connect_button.config(text="Connect Dobot", state=tk.NORMAL)
        return False


#Function to disconnect and reconnect Dobot
def disconnect_reconnect_dobot():
    global dobot, connected
    update_chat("Mike: Attempting to disconnect and reconnect...", "system")
    reconnect_button.config(text="Reconnecting...", state=tk.DISABLED)
    root.update()

    # Disconnect
    if dobot is not None:
        try:
            dobot.close()
        except:
            pass
        dobot = None
        connected = False

    # Wait a moment
    time.sleep(2)

    # Reconnect
    result = connect_dobot()
    reconnect_button.config(text="Reconnect Dobot", state=tk.NORMAL)
    return result


# Function to update camera feed in normal mode
def update_camera():
    global mode

    if mode != "normal":
        return

    try:
        ret, frame = cap.read()
        if ret:
            # Add text overlay showing current mode
            cv2.putText(frame, "Mode: Normal", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert OpenCV BGR to RGB
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            camera_label.imgtk = imgtk
            camera_label.config(image=imgtk)
    except Exception as e:
        # In case of camera error, just continue
        pass

    # Continue updating camera feed if in normal mode
    if mode == "normal":
        camera_label.after(33, update_camera)  # ~30 FPS


# Function for object recognition (placeholder)     """""add here"""""""

def object_recognition_mode():
    """
    Object recognition mode using YOLO with the laptop's webcam.
    The processed frame is displayed in the GUI's camera feed (camera_label).
    Every 3 seconds, the detected objects (from YOLO) are directly shown as the LLM response in the chat.
    """
    global mode
    from ultralytics import YOLO
    from Modules.camera_stream import CameraStream
    from Modules.speech import speak
    from Modules.color_detection import get_color_name
    import time
    import cv2
    from PIL import Image, ImageTk

    CAMERA_SRC = 0  # Use the default webcam
    model = YOLO("models/yolov8n.pt")  # Ensure your model file exists
    cap = CameraStream(CAMERA_SRC)
    FRAME_WIDTH = int(cap.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = 0
    last_detection_time = time.time()
    last_llm_time = time.time()

    # List of objects to detect
    objects_to_detect = [
        "person", "cell phone", "chair", "door", "cup", "bottle", "laptop",
        "fruit", "apple", "banana", "orange", "grape", "strawberry", "pineapple",
        "wall", "tv", "flower", "book", "mirror", "keyboard", "mouse", "table",
        "bed", "sofa", "fan", "light", "clock", "monitor", "remote", "microwave",
        "car", "motorcycle", "bicycle", "truck", "bus", "helmet", "backpack", "umbrella"
    ]

    update_chat("Mike: Object recognition mode activated using YOLO and webcam.", "mike")
    mode = "object"

    while cap.running and mode == "object":
        ret, frame = cap.read()
        if not ret:
            update_chat("Mike: Lost connection to camera. Retrying...", "system")
            cap.stop()
            cap = CameraStream(CAMERA_SRC)
            continue

        frame_count += 1
        # Process every 3rd frame for efficiency; update the GUI regardless
        if frame_count % 3 != 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            camera_label.config(image=imgtk)
            camera_label.imgtk = imgtk
            time.sleep(0.03)
            continue

        results = model(frame, imgsz=320, conf=0.5, half=True)
        detected_objects = []

        for r in results:
            for i, box in enumerate(r.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box[:4])
                class_id = int(r.boxes.cls[i])
                label = model.names.get(class_id, "Unknown")
                if label in objects_to_detect:
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    if center_x < FRAME_WIDTH // 3:
                        direction = "on your left"
                    elif center_x > 2 * FRAME_WIDTH // 3:
                        direction = "on your right"
                    else:
                        direction = "in front of you"
                    if center_y < FRAME_HEIGHT // 3:
                        direction += ", at the top"
                    elif center_y > 2 * FRAME_HEIGHT // 3:
                        direction += ", at the bottom"
                    if label == "person":
                        detected_objects.append(f"{label} {direction}")
                        annotation = f"{label}, {direction}"
                    else:
                        object_roi = frame[y1:y2, x1:x2]
                        avg_color = object_roi.mean(axis=(0, 1))
                        detected_color = get_color_name(avg_color[0], avg_color[1], avg_color[2])
                        detected_objects.append(f"{label} {detected_color} {direction}")
                        annotation = f"{label}, {detected_color}, {direction}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, annotation, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- LLM Interaction Block ---
        # Instead of calling the external LLM, we now directly use the YOLO output as the LLM response.
        if detected_objects and (time.time() - last_llm_time > 3):
            objects_str = ", ".join(detected_objects)
            update_chat(f"LLM: {objects_str}", "mike")
            last_llm_time = time.time()

        # Use text-to-speech every 3 seconds if objects are detected
        if detected_objects and (time.time() - last_detection_time > 3):
            speak("I see " + ", ".join(detected_objects))
            last_detection_time = time.time()

        # Update the GUI camera feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.config(image=imgtk)
        camera_label.imgtk = imgtk
        time.sleep(0.03)

    cap.stop()
    update_camera()  # Resume the normal camera feed

def activate_object_recognition_mode():
    global mode
    # Do not allow object recognition mode if the Dobot is not connected.
    if not connected:
        update_chat("Mike: Please connect the Dobot before activating object recognition mode.", "error")
        return
    mode = "object"
    threading.Thread(target=object_recognition_mode, daemon=True).start()


# Function to extract shape and dimensions using Gemini
def get_shape_from_gemini(user_input):
    prompt = (
        f"Extract the shape and dimensions from this request: '{user_input}'. "
        "Respond only with a JSON object in this format:\n"
        "{\"shape\": \"circle\", \"dimensions\": [5]}"
    )

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        clean_text = re.sub(r"```json\n|\n```", "", raw_text)

        shape_data = json.loads(clean_text)
        if "shape" in shape_data and "dimensions" in shape_data:
            extracted_info = f"I will draw a {shape_data['shape']} with dimensions {shape_data['dimensions']}! 🎨"
            update_chat(f"Mike: {extracted_info}", "mike")

            # Now draw the shape with Dobot
            threading.Thread(target=draw_shape_with_dobot, args=(shape_data["shape"], shape_data["dimensions"]),
                             daemon=True).start()
        else:
            update_chat("Mike: Error in shape extraction.", "error")
    except Exception as e:
        update_chat(f"Mike: Failed to extract shape data: {str(e)}", "error")


def draw_shape_with_dobot(shape, dimensions):
    global dobot, connected

    if not connected:
        if not connect_dobot():
            return

    try:
        # Start from home position and prepare to draw
        update_chat("Mike: Moving to start position...", "system")
        dobot.move_to(home_x, home_y, home_z, 0, wait=True)  # Move to home position

        # Show what we're about to draw in the chat
        shape_name = shape.lower()
        if shape_name == "circle":
            radius = dimensions[0]
            update_chat(f"Mike: I'm drawing a circle with radius {radius}. Hold on! 🖌️", "mike")
        elif shape_name == "square":
            side = dimensions[0]
            update_chat(f"Mike: I'm drawing a square with side length {side}. This is fun! 🖌️", "mike")
        elif shape_name == "rectangle":
            length = dimensions[0]
            width = dimensions[1] if len(dimensions) > 1 else dimensions[0]
            update_chat(f"Mike: I'm drawing a rectangle with length {length} and width {width}. Watch this! 🖌️", "mike")
        elif shape_name == "triangle":
            side = dimensions[0]
            update_chat(f"Mike: I'm drawing a triangle with side length {side}. Here we go! 🖌️", "mike")
        elif shape_name == "line":
            length = dimensions[0]
            update_chat(f"Mike: I'm drawing a line with length {length}. Simple but elegant! 🖌️", "mike")
        else:
            update_chat(f"Mike: I'll try to draw a {shape_name}. Let's see how it goes! 🖌️", "mike")

        # Wait a moment for the chat messages to be visible
        time.sleep(1)

        # Lower pen to drawing position
        dobot.move_to(home_x, home_y, drawing_z, 0, wait=True)

        shape = shape.lower()

        if shape == "circle":
            draw_circle(dimensions[0])
        elif shape == "square" or shape == "rectangle":
            if len(dimensions) == 1:  # Square
                draw_square(dimensions[0])
            elif len(dimensions) >= 2:  # Rectangle
                draw_rectangle(dimensions[0], dimensions[1])
        elif shape == "triangle":
            if len(dimensions) == 1:  # Equilateral triangle
                draw_triangle(dimensions[0])
            elif len(dimensions) == 2:  # Isosceles triangle
                draw_triangle(dimensions[0], dimensions[1])
            elif len(dimensions) >= 3:  # Scalene triangle
                draw_triangle(dimensions[0], dimensions[1], dimensions[2])
        elif shape == "line":
            if len(dimensions) >= 1:
                draw_line(dimensions[0])
        else:
            update_chat(f"Mike: Sorry, I don't know how to draw a {shape} yet.", "error")

        # Return to home position
        dobot.move_to(home_x, home_y, home_z, 0, wait=True)
        update_chat("Mike: Drawing completed! ✅", "system")
        update_chat("Mike: How does it look? Would you like me to draw something else? 🎨", "mike")

    except Exception as e:
        update_chat(f"Mike: Error during drawing: {str(e)}", "error")
        # Try to reconnect if there's a connection issue
        if "NoneType" in str(e) or "params" in str(e) or "timeout" in str(e).lower():
            update_chat("Mike: Connection issue detected. Attempting to reconnect...", "system")
            threading.Thread(target=disconnect_reconnect_dobot, daemon=True).start()


# Shape drawing functions
def draw_circle(radius):
    global dobot

    #Convert radius to robot units (mm)
    radius = float(radius) * 10  # Assume input is in cm, convert to mm

    # Define circle parameters
    center_x = home_x
    center_y = home_y
    steps = 36  # Number of segments to approximate the circle can increase for more preices but slower drawing

    # Lower pen to paper
    dobot.move_to(center_x + radius, center_y, drawing_z, 0, wait=True)

    # Draw the circle
    for i in range(steps + 1):
        angle = 2 * math.pi * i / steps
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        dobot.move_to(x, y, drawing_z, 0, wait=True)

    # Lift pen
    dobot.move_to(center_x + radius, center_y, home_z, 0, wait=True)


def draw_square(side_length):
    global dobot

    # Convert side length to robot units (mm)
    side = float(side_length) * 10  # Assume input is in cm, convert to mm

    # Calculate starting position (center the square)
    start_x = home_x - side / 2
    start_y = home_y - side / 2

    # Move to starting position and lower pen
    dobot.move_to(start_x, start_y, home_z, 0, wait=True)
    dobot.move_to(start_x, start_y, drawing_z, 0, wait=True)

    # Draw square
    dobot.move_to(start_x + side, start_y, drawing_z, 0, wait=True)
    dobot.move_to(start_x + side, start_y + side, drawing_z, 0, wait=True)
    dobot.move_to(start_x, start_y + side, drawing_z, 0, wait=True)
    dobot.move_to(start_x, start_y, drawing_z, 0, wait=True)

    # Lift pen
    dobot.move_to(start_x, start_y, home_z, 0, wait=True)


def draw_rectangle(length, width):
    global dobot

    # Convert dimensions to robot units (mm)
    length = float(length) * 10  # Assume input is in cm, convert to mm
    width = float(width) * 10

    # Calculate starting position (center the rectangle)
    start_x = home_x - length / 2
    start_y = home_y - width / 2

    # Move to starting position and lower pen
    dobot.move_to(start_x, start_y, home_z, 0, wait=True)
    dobot.move_to(start_x, start_y, drawing_z, 0, wait=True)

    # Draw rectangle
    dobot.move_to(start_x + length, start_y, drawing_z, 0, wait=True)
    dobot.move_to(start_x + length, start_y + width, drawing_z, 0, wait=True)
    dobot.move_to(start_x, start_y + width, drawing_z, 0, wait=True)
    dobot.move_to(start_x, start_y, drawing_z, 0, wait=True)

    # Lift pen
    dobot.move_to(start_x, start_y, home_z, 0, wait=True)


def draw_triangle(side1, side2=None, side3=None):
    global dobot

    if side2 is None and side3 is None:
        # Draw equilateral triangle
        side = float(side1) * 10  # Convert to mm

        # Calculate height of equilateral triangle
        height = side * math.sqrt(3) / 2

        # Calculate starting position
        start_x = home_x - side / 2
        start_y = home_y - height / 3  # Offset to center

        # Move to starting position and lower pen
        dobot.move_to(start_x, start_y, home_z, 0, wait=True)
        dobot.move_to(start_x, start_y, drawing_z, 0, wait=True)

        # Draw triangle
        dobot.move_to(start_x + side, start_y, drawing_z, 0, wait=True)
        dobot.move_to(start_x + side / 2, start_y + height, drawing_z, 0, wait=True)
        dobot.move_to(start_x, start_y, drawing_z, 0, wait=True)

        # Lift pen
        dobot.move_to(start_x, start_y, home_z, 0, wait=True)

    elif side3 is None:
        # Draw isosceles triangle (two sides equal)
        side_a = float(side1) * 10  # Convert to mm
        side_b = float(side2) * 10  # Convert to mm

        # Use side_a as the base and side_b as the equal sides
        base = side_a
        equal_sides = side_b

        # Calculate height using Pythagorean theorem
        half_base = base / 2
        height = math.sqrt(equal_sides ** 2 - half_base ** 2)

        # Calculate starting position
        start_x = home_x - base / 2
        start_y = home_y - height / 3  # Offset to center

        # Move to starting position and lower pen
        dobot.move_to(start_x, start_y, home_z, 0, wait=True)
        dobot.move_to(start_x, start_y, drawing_z, 0, wait=True)

        # Draw triangle
        dobot.move_to(start_x + base, start_y, drawing_z, 0, wait=True)
        dobot.move_to(start_x + base / 2, start_y + height, drawing_z, 0, wait=True)
        dobot.move_to(start_x, start_y, drawing_z, 0, wait=True)

        # Lift pen
        dobot.move_to(start_x, start_y, home_z, 0, wait=True)

    else:
        # Draw scalene triangle (all sides different)
        side_a = float(side1) * 10  # Convert to mm
        side_b = float(side2) * 10  # Convert to mm
        side_c = float(side3) * 10  # Convert to mm

        # Check if triangle is valid using triangle inequality theorem
        if (side_a + side_b <= side_c) or (side_a + side_c <= side_b) or (side_b + side_c <= side_a):
            update_chat(
                "Mike: Invalid triangle dimensions. The sum of any two sides must be greater than the third side.",
                "system")
            return

        # Use the Law of Cosines to calculate the angles
        angle_a = math.acos((side_b ** 2 + side_c ** 2 - side_a ** 2) / (2 * side_b * side_c))
        angle_b = math.acos((side_a ** 2 + side_c ** 2 - side_b ** 2) / (2 * side_a * side_c))
        angle_c = math.pi - angle_a - angle_b  # Angles in a triangle sum to π radians

        # Calculate coordinates using side_c as the base
        start_x = home_x - side_c / 2
        start_y = home_y

        # Second point (end of base)
        second_x = start_x + side_c
        second_y = start_y

        # Third point (using Law of Cosines)
        third_x = start_x + side_b * math.cos(angle_c)
        third_y = start_y + side_b * math.sin(angle_c)

        # Move to starting position and lower pen
        dobot.move_to(start_x, start_y, home_z, 0, wait=True)
        dobot.move_to(start_x, start_y, drawing_z, 0, wait=True)

        # Draw triangle
        dobot.move_to(second_x, second_y, drawing_z, 0, wait=True)
        dobot.move_to(third_x, third_y, drawing_z, 0, wait=True)
        dobot.move_to(start_x, start_y, drawing_z, 0, wait=True)

        # Lift pen
        dobot.move_to(start_x, start_y, home_z, 0, wait=True)


def draw_line(length):
    global dobot

    # Convert length to robot units (mm)
    length = float(length) * 10  # Assume input is in cm, convert to mm

    # Calculate starting position
    start_x = home_x - length / 2
    start_y = home_y

    # Move to starting position and lower pen
    dobot.move_to(start_x, start_y, home_z, 0, wait=True)
    dobot.move_to(start_x, start_y, drawing_z, 0, wait=True)

    # Draw line
    dobot.move_to(start_x + length, start_y, drawing_z, 0, wait=True)

    # Lift pen
    dobot.move_to(start_x + length, start_y, home_z, 0, wait=True)


# Function to check if dobot is responsive
def check_dobot_connection():
    global dobot, connected
    if not connected or dobot is None:
        return False

    try:
        # Try to get the current position to check if connection is alive
        (x, y, z, r, j1, j2, j3, j4) = dobot.pose()
        return True
    except Exception as e:
        update_chat(f"Mike: Connection check failed: {str(e)}", "system")
        return False


# Function to mix colors using Gemini
def mix_colors(user_input):
    prompt = (
        f"I want to create the color '{user_input}' using only red, blue, and green (RGB model). "
        "Tell me which of these three colors I should mix to get the closest possible result. "
        "Respond only with a JSON object in this format:\n"
        "{\"mix\": [\"red\", \"blue\"]}"
    )

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        clean_text = re.sub(r"```json\n|\n```", "", raw_text)

        color_data = json.loads(clean_text)
        if "mix" in color_data:
            colors = color_data["mix"]
            color_text = ", ".join(colors)
            update_chat(f"Mike: To create {user_input}, I would mix {color_text}! 🎨", "mike")
        else:
            update_chat("Mike: Error in color extraction.", "error")

    except Exception as e:
         update_chat(f"Mike: Failed to extract color data: {str(e)}", "error")


# Function to move dobot to home position
def go_home():
    global dobot, connected, home_x, home_y, home_z

    if not connected:
        update_chat("Mike: Please connect the Dobot first!", "error")
        return

    try:
        update_chat("Mike: Moving to home position... 🏠", "system")
        dobot.move_to(home_x, home_y, home_z, 0, wait=True)
        update_chat(f"Mike: I'm home! Position: X={home_x}, Y={home_y}, Z={home_z}", "mike")
    except Exception as e:
        update_chat(f"Mike: Error moving to home: {str(e)}", "error")
        # Try to reconnect if there's a connection issue
        if "NoneType" in str(e) or "params" in str(e) or "timeout" in str(e).lower():
            threading.Thread(target=disconnect_reconnect_dobot, daemon=True).start()


# Function to handle speech recognition
def voice_command():
    global mode

    # If we're in gesture mode, exit it
    if mode == "gesture":
        deactivate_gesture_mode()
        return

    update_chat("Mike: Listening... 🎤", "system")

    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=5, phrase_time_limit=5)

        text = r.recognize_google(audio)
        update_chat(f"You: {text}", "user")

        # Check for gesture mode activation/deactivation
        if "gesture mode" in text.lower() or "hand control" in text.lower():
            activate_gesture_mode()
            return
        elif "exit gesture" in text.lower() or "quit gesture" in text.lower():
            deactivate_gesture_mode()
            return

        # Check for manual control mode
        if "manual control" in text.lower() or "manual mode" in text.lower():
            activate_manual_control()
            return

        # Check for shape drawing commands
        if "draw" in text.lower() or "make" in text.lower():
            get_shape_from_gemini(text)
            return

        # Check for other common commands
        if "home" in text.lower():
            threading.Thread(target=go_home, daemon=True).start()
            return
        elif "connect" in text.lower():
            threading.Thread(target=connect_dobot, daemon=True).start()
            return
        elif "disconnect" in text.lower() or "reconnect" in text.lower():
            threading.Thread(target=disconnect_reconnect_dobot, daemon=True).start()
            return

        # Default action for unrecognized commands
        process_nlp_command(text)
    except sr.WaitTimeoutError:
        update_chat("Mike: Listening timed out. Please try again.", "system")
    except sr.UnknownValueError:
        update_chat("Mike: Sorry, I couldn't understand what you said.", "system")
    except sr.RequestError as e:
        update_chat(f"Mike: Speech recognition service error: {str(e)}", "error")
    except Exception as e:
        update_chat(f"Mike: Error processing voice command: {str(e)}", "error")


# Function to process natural language commands using Gemini
def process_nlp_command(text):
    prompt = (
        f"I am Mike, a robot assistant that controls a Dobot robotic arm. The user has told me: '{text}'. "
        "Based on this input, determine what action I should take. "
        "Respond with ONLY ONE of these exact commands (no additional text):\n"
        "1. DRAW_SHAPE if they want me to draw something\n"
        "2. GO_HOME if they want me to return to home position\n"
        "3. CONNECT if they want me to connect\n"
        "4. RECONNECT if they mention reconnection\n"
        "5. GESTURE_MODE if they want hand gesture control\n"
        "6. MANUAL_MODE if they want manual control\n"
        "7. UNKNOWN if I can't determine a specific action\n"
    )

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        command = response.text.strip()

        if command == "DRAW_SHAPE":
            get_shape_from_gemini(text)
        elif command == "GO_HOME":
            threading.Thread(target=go_home, daemon=True).start()
        elif command == "CONNECT":
            threading.Thread(target=connect_dobot, daemon=True).start()
        elif command == "RECONNECT":
            threading.Thread(target=disconnect_reconnect_dobot, daemon=True).start()
        elif command == "GESTURE_MODE":
            activate_gesture_mode()
        elif command == "MANUAL_MODE":
            activate_manual_control()
        else:  # UNKNOWN or any other response
            update_chat(
                "Mike: I'm not sure what you want me to do. Try asking me to draw a shape, go home, or activate gesture mode.",
                "mike")
    except Exception as e:
        update_chat(f"Mike: Error processing command: {str(e)}", "error")


# Function to update chat display
def update_chat(message, message_type="user"):
    chat_display.config(state=tk.NORMAL)

    # Set tag based on message type
    if message_type == "user":
        tag = "user_msg"
        chat_display.insert(tk.END, f"{message}\n\n", tag)
    elif message_type == "mike":
        tag = "mike_msg"
        chat_display.insert(tk.END, f"{message}\n\n", tag)
    elif message_type == "system":
        tag = "system_msg"
        chat_display.insert(tk.END, f"{message}\n", tag)
    elif message_type == "error":
        tag = "error_msg"
        chat_display.insert(tk.END, f"{message}\n", tag)

    chat_display.config(state=tk.DISABLED)
    chat_display.see(tk.END)  # Scroll to the end


# Function to handle sending messages
def send_message(event=None):
    message = input_field.get()
    if message.strip() == "":
        return

    update_chat(f"You: {message}", "user")
    input_field.delete(0, tk.END)

    # Check for exit commands
    if message.lower() in ["quit", "exit", "stop"]:
        if mode == "gesture":
            deactivate_gesture_mode()
            return
        elif mode == "manual":
            deactivate_manual_control()
            return

    # Check for manual control mode
    if "manual" in message.lower() and ("control" in message.lower() or "mode" in message.lower()):
        activate_manual_control()
        return

    # Check for gesture control mode
    if "gesture" in message.lower() and ("control" in message.lower() or "mode" in message.lower()):
        activate_gesture_mode()
        return

    # Process message for shape drawing or other commands
    if "draw" in message.lower() or "make" in message.lower():
        get_shape_from_gemini(message)
    else:
        process_nlp_command(message)


# Manual control mode functions
def activate_manual_control():
    global mode

    if not connected:
        update_chat("Mike: Please connect the Dobot first before activating manual control mode.", "error")
        return

    mode = "manual"
    update_chat("Mike: Manual control mode activated! ⌨️ Use the arrow keys and buttons below to control the robot.",
                "mike")

    # Show manual control panel if not already visible
    show_manual_controls()


def deactivate_manual_control():
    global mode

    mode = "normal"
    update_chat("Mike: Manual control mode deactivated. Back to normal operation! 🤖", "mike")

    # Hide manual control panel
    hide_manual_controls()


def show_manual_controls():
    manual_control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)


def hide_manual_controls():
    manual_control_frame.pack_forget()


# Functions for manual movement
def move_x_plus():
    if check_dobot_connection():
        x, y, z, r, *_ = dobot.pose()
        new_x = min(x + 10, workspace_limits["x_max"])
        dobot.move_to(new_x, y, z, r, wait=True)
        update_chat(f"Mike: Moving X+: {new_x:.1f}", "system")


def move_x_minus():
    if check_dobot_connection():
        x, y, z, r, *_ = dobot.pose()
        new_x = max(x - 10, workspace_limits["x_min"])
        dobot.move_to(new_x, y, z, r, wait=True)
        update_chat(f"Mike: Moving X-: {new_x:.1f}", "system")


def move_y_plus():
    if check_dobot_connection():
        x, y, z, r, *_ = dobot.pose()
        new_y = min(y + 10, workspace_limits["y_max"])
        dobot.move_to(x, new_y, z, r, wait=True)
        update_chat(f"Mike: Moving Y+: {new_y:.1f}", "system")


def move_y_minus():
    if check_dobot_connection():
        x, y, z, r, *_ = dobot.pose()
        new_y = max(y - 10, workspace_limits["y_min"])
        dobot.move_to(x, new_y, z, r, wait=True)
        update_chat(f"Mike: Moving Y-: {new_y:.1f}", "system")


def move_z_plus():
    if check_dobot_connection():
        x, y, z, r, *_ = dobot.pose()
        new_z = min(z + 10, workspace_limits["z_max"])
        dobot.move_to(x, y, new_z, r, wait=True)
        update_chat(f"Mike: Moving Z+: {new_z:.1f}", "system")


def move_z_minus():
    if check_dobot_connection():
        x, y, z, r, *_ = dobot.pose()
        new_z = max(z - 10, workspace_limits["z_min"])
        dobot.move_to(x, y, new_z, r, wait=True)
        update_chat(f"Mike: Moving Z-: {new_z:.1f}", "system")


def toggle_suction():
    global suction_state
    if check_dobot_connection():
        suction_state = not suction_state
        dobot.suck(suction_state)
        status = "ON" if suction_state else "OFF"
        update_chat(f"Mike: Suction {status}", "system")
        suction_button.config(text=f"Suction {'OFF' if suction_state else 'ON'}")


# Main application setup
root = tk.Tk()
root.title("Mike - Dobot AI Control Assistant")
root.geometry("1200x800")
root.configure(bg="#f0f0f0")

# Set icon (if available)
try:
    icon_img = Image.open("robot_icon.png")
    icon_photo = ImageTk.PhotoImage(icon_img)
    root.iconphoto(True, icon_photo)
except:
    pass  # Skip if icon file is not available

# Global state variables
suction_state = False
mode = "normal"

# Create a style
style = ttk.Style()
style.configure("TButton", padding=6, font=('Segoe UI', 10))
style.configure("TFrame", background="#f0f0f0")
style.configure("TLabel", background="#f0f0f0", font=('Segoe UI', 10))

# Create main frames
left_frame = ttk.Frame(root, padding="10")
right_frame = ttk.Frame(root, padding="10")

left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Left frame - Chat interface
chat_frame = ttk.Frame(left_frame)
chat_frame.pack(fill=tk.BOTH, expand=True, pady=5)

# Title for chat section
chat_title = ttk.Label(chat_frame, text="Mike - Robot Assistant", font=('Segoe UI', 16, 'bold'))
chat_title.pack(pady=5)

# Chat display
chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, width=40, height=20, font=('Segoe UI', 10))
chat_display.pack(fill=tk.BOTH, expand=True, pady=5)
chat_display.tag_configure("user_msg", foreground="#0078D7", font=('Segoe UI', 10, 'bold'))
chat_display.tag_configure("mike_msg", foreground="#009933", font=('Segoe UI', 10))
chat_display.tag_configure("system_msg", foreground="#6B6B6B", font=('Segoe UI', 9, 'italic'))
chat_display.tag_configure("error_msg", foreground="#FF0000", font=('Segoe UI', 9, 'bold'))
chat_display.config(state=tk.DISABLED)

# Input field and send button frame
input_frame = ttk.Frame(left_frame)
input_frame.pack(fill=tk.X, pady=5)

input_field = ttk.Entry(input_frame, font=('Segoe UI', 10))
input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
input_field.bind("<Return>", send_message)

send_button = ttk.Button(input_frame, text="Send", command=send_message)
send_button.pack(side=tk.RIGHT)

# Control buttons frame
control_frame = ttk.Frame(left_frame)
control_frame.pack(fill=tk.X, pady=5)

connect_button = ttk.Button(control_frame, text="Connect Dobot",
                            command=lambda: threading.Thread(target=connect_dobot, daemon=True).start())
connect_button.pack(side=tk.LEFT, padx=5)

reconnect_button = ttk.Button(control_frame, text="Reconnect Dobot",
                              command=lambda: threading.Thread(target=disconnect_reconnect_dobot, daemon=True).start(),
                              state=tk.DISABLED)
reconnect_button.pack(side=tk.LEFT, padx=5)

home_button = ttk.Button(control_frame, text="Go Home",
                         command=lambda: threading.Thread(target=go_home, daemon=True).start(), state=tk.DISABLED)
home_button.pack(side=tk.LEFT, padx=5)

voice_button = ttk.Button(control_frame, text="🎤 Speak",
                          command=lambda: threading.Thread(target=voice_command, daemon=True).start())
voice_button.pack(side=tk.LEFT, padx=5)

# Right frame - Camera feed and controls
camera_frame = ttk.Frame(right_frame)
camera_frame.pack(fill=tk.BOTH, expand=True, pady=5)

# Title for camera section
camera_title = ttk.Label(camera_frame, text="Camera Feed", font=('Segoe UI', 16, 'bold'))
camera_title.pack(pady=5)

# Camera feed display
camera_label = ttk.Label(camera_frame)
camera_label.pack(fill=tk.BOTH, expand=True, pady=5)

# Mode buttons frame
mode_frame = ttk.Frame(right_frame)
mode_frame.pack(fill=tk.X, pady=5)

gesture_button = ttk.Button(mode_frame, text="Hand Gesture Mode",
                            command=lambda: threading.Thread(target=activate_gesture_mode, daemon=True).start())
gesture_button.pack(side=tk.LEFT, padx=5)

manual_button = ttk.Button(mode_frame, text="Manual Control Mode",
                           command=lambda: threading.Thread(target=activate_manual_control, daemon=True).start())
manual_button.pack(side=tk.LEFT, padx=5)

object_button = ttk.Button(mode_frame, text="Object Recognition Mode",
                           command=lambda: threading.Thread(target=activate_object_recognition_mode, daemon=True).start())
object_button.pack(side=tk.LEFT, padx=5)

normal_button = ttk.Button(mode_frame, text="Normal Mode", command=lambda: threading.Thread(
    target=deactivate_gesture_mode if mode == "gesture" else deactivate_manual_control, daemon=True).start())
normal_button.pack(side=tk.LEFT, padx=5)

# Manual control frame (hidden by default)
manual_control_frame = ttk.Frame(right_frame)

# Create a 3x3 grid for movement buttons
movement_grid = ttk.Frame(manual_control_frame)
movement_grid.pack(pady=10)

# Z-axis controls (top row)
z_up_button = ttk.Button(movement_grid, text="Z+",
                         command=lambda: threading.Thread(target=move_z_plus, daemon=True).start())
z_up_button.grid(row=0, column=1, padx=5, pady=5)

# Spacer
ttk.Label(movement_grid, text="").grid(row=0, column=0, padx=5, pady=5)
ttk.Label(movement_grid, text="").grid(row=0, column=2, padx=5, pady=5)

# X/Y controls (middle row)
y_minus_button = ttk.Button(movement_grid, text="Y-",
                            command=lambda: threading.Thread(target=move_y_minus, daemon=True).start())
y_minus_button.grid(row=1, column=0, padx=5, pady=5)

home_grid_button = ttk.Button(movement_grid, text="Home",
                              command=lambda: threading.Thread(target=go_home, daemon=True).start())
home_grid_button.grid(row=1, column=1, padx=5, pady=5)

y_plus_button = ttk.Button(movement_grid, text="Y+",
                           command=lambda: threading.Thread(target=move_y_plus, daemon=True).start())
y_plus_button.grid(row=1, column=2, padx=5, pady=5)

# Z-axis controls (bottom row)
z_down_button = ttk.Button(movement_grid, text="Z-",
                           command=lambda: threading.Thread(target=move_z_minus, daemon=True).start())
z_down_button.grid(row=2, column=1, padx=5, pady=5)

# X-axis controls
x_minus_button = ttk.Button(movement_grid, text="X-",
                            command=lambda: threading.Thread(target=move_x_minus, daemon=True).start())
x_minus_button.grid(row=2, column=0, padx=5, pady=5)

x_plus_button = ttk.Button(movement_grid, text="X+",
                           command=lambda: threading.Thread(target=move_x_plus, daemon=True).start())
x_plus_button.grid(row=2, column=2, padx=5, pady=5)

# Suction control
suction_button = ttk.Button(manual_control_frame, text="Suction ON", command=toggle_suction)
suction_button.pack(pady=10)

# Exit manual control button
exit_manual_button = ttk.Button(manual_control_frame, text="Exit Manual Mode", command=deactivate_manual_control)
exit_manual_button.pack(pady=5)

# Initialize camera
cap = cv2.VideoCapture(0)  # Change to appropriate camera index if needed
update_camera()

# Welcome message
update_chat(
    "Mike: Hello! I'm Mike, your robotic assistant. I can help you control the Dobot Magician robot arm.\n"
    "You can ask me to draw shapes, connect to the robot, or use gesture control mode.\n"
    "To get started, please connect the Dobot using the 'Connect Dobot' button!", "mike"
)


# Keyboard shortcuts for manual control
def on_key_press(event):
    if mode == "manual":
        key = event.keysym.lower()
        if key == "up":
            threading.Thread(target=move_y_plus, daemon=True).start()
        elif key == "down":
            threading.Thread(target=move_y_minus, daemon=True).start()
        elif key == "left":
            threading.Thread(target=move_x_minus, daemon=True).start()
        elif key == "right":
            threading.Thread(target=move_x_plus, daemon=True).start()
        elif key == "w":
            threading.Thread(target=move_z_plus, daemon=True).start()
        elif key == "s":
            threading.Thread(target=move_z_minus, daemon=True).start()
        elif key == "space":
            threading.Thread(target=toggle_suction, daemon=True).start()
        elif key == "h":
            threading.Thread(target=go_home, daemon=True).start()
        elif key == "escape":
            deactivate_manual_control()


root.bind("<Key>", on_key_press)


# Clean up when closing
def on_closing():
    global dobot, stop_gesture_thread

    # Stop gesture thread if running
    stop_gesture_thread.set()

    # Release camera
    if cap.isOpened():
        cap.release()

    # Disconnect from Dobot
    if dobot is not None:
        try:
            dobot.close()
        except:
            pass

    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the main loop
root.mainloop()