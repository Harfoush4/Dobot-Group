import tkinter as tk
from tkinter import messagebox
import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Function to handle speech-to-text
def listen_and_transcribe():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)  # Use Google Web Speech API
            text_box.delete(1.0, tk.END)  # Clear previous text
            text_box.insert(tk.END, text)  # Insert new text
        except sr.UnknownValueError:
            messagebox.showerror("Error", "Sorry, I could not understand the audio.")
        except sr.RequestError:
            messagebox.showerror("Error", "Could not request results; check your internet connection.")

# Create the GUI
root = tk.Tk()
root.title("Speech to Text")

# Add a button to start listening
listen_button = tk.Button(root, text="Press to Speak", command=listen_and_transcribe)
listen_button.pack(pady=20)

# Add a text box to display the transcribed text
text_box = tk.Text(root, height=10, width=50)
text_box.pack(pady=10)

# Run the app
root.mainloop()