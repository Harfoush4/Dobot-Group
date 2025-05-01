# Mike - AI-Powered Dobot Magician Controller

## Overview
**Mike** is an intelligent assistant designed to control the Dobot Magician robotic arm using multiple interfaces including:

- **Natural Language Commands** (via Speech or Text)
- **Hand Gesture Recognition**
- **Manual Keyboard and Button Control**
- **Object Recognition with YOLOv8**

It uses a combination of computer vision (MediaPipe, OpenCV), speech recognition, Google Gemini AI (LLM), and YOLOv8 object detection for a multimodal control experience.

---

## Features
- 🔊 **Voice Interaction** with Gemini LLM for interpreting tasks
- 👋 **Hand Gesture Mode**: Control robot arm via camera-tracked hand gestures
- 🎭 **Shape Drawing**: Ask Mike to draw shapes like circles, squares, triangles
- 👀 **YOLO-based Object Recognition** for describing environment
- ⌨️ **Manual Control Panel** for traditional movement
- 💡 **Color Mixing**: Ask Mike how to mix RGB colors to form a target color
- 🔜 Automatic switching between control modes

---

## Technologies Used
| Category             | Tech Stack                              |
|----------------------|------------------------------------------|
| GUI Framework        | Tkinter                                  |
| Computer Vision      | OpenCV, MediaPipe, PIL                   |
| Object Detection     | YOLOv8 via Ultralytics                   |
| LLM Integration      | Google Gemini API                        |
| Speech Recognition   | `speech_recognition` library             |
| Robot Control        | pydobot (Dobot Magician)                 |
| Multithreading       | Python `threading` and `queue`           |

---

## Installation
1. Clone the repository:
```bash
git clone https://github.com/your-repo/mike-dobot-assistant.git
cd mike-dobot-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your **Google Gemini API key**:
```python
# Replace in the code
genai.configure(api_key="YOUR_GEMINI_API_KEY")
```

4. Plug in and power your **Dobot Magician** via USB.

5. Run the application:
```bash
python main.py
```

---

## Requirements
- Python 3.8+
- Dobot Magician + pydobot
- Webcam (for gesture and object recognition)
- Google Gemini API access
- Optional: microphone for voice commands

---

## How to Use
1. **Connect** the Dobot using the "Connect Dobot" button.
2. Use the chatbox or microphone to command Mike:
   - "Draw a square of 5 cm"
   - "Enter gesture control mode"
   - "Go home"
3. Activate **Gesture Mode** and use:
   - ✋ "V" gesture: Suction ON
   - 🎒 "Gun" gesture: Suction OFF
   - Point to move in X/Y/Z
4. Use **Manual Mode** for precise control via GUI or keyboard.
5. Activate **Object Recognition Mode** to see what's around.

---

## Voice Commands Examples
| Phrase                         | Action                       |
|-------------------------------|------------------------------|
| "Connect to the robot"        | Connects Dobot               |
| "Draw a triangle of 6 cm"     | Initiates drawing            |
| "Gesture mode"                | Enables hand gesture control |
| "Exit gesture mode"           | Returns to normal            |
| "Go home"                     | Moves robot to home position |

---

## Folder Structure (Example)
```
Mike-Dobot-Assistant/
├── models/
│   └── yolov8n.pt
├── Modules/
│   ├── camera_stream.py
│   ├── speech.py
│   └── color_detection.py
├── main.py
├── README.md
└── requirements.txt
```

---

## Known Issues
- Real-time gesture detection may vary under low-light conditions
- Gemini API has rate limits (adjust interaction frequency if needed)
- Object recognition requires good lighting and proper webcam

---

## License
MIT License

---

## Credits
Developed by **ARAS** as part of an intelligent robotic control system for Dobot integration with AI and computer vision.

---

## Future Work
- Add voice feedback using TTS
- Export drawn shapes as SVG or DXF
- Multi-language support
- More complex gesture library

