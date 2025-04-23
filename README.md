✍️ Math with Gestures – AI Handwriting Recognition with Streamlit & CVZone


This project allows users to draw mathematical expressions using hand gestures in real-time using a webcam. Leveraging OpenCV, CVZone's HandTrackingModule, and Streamlit, the system detects hand landmarks to capture gesture-based writing, processes the drawn input, and sends it to an AI model for interpretation and solving.

🔍 Features
Real-time hand tracking using webcam

Gesture-based drawing on a virtual canvas

Erase canvas using a specific finger gesture

AI integration to solve handwritten math expressions

Web interface built with Streamlit for accessibility

🛠️ Technologies Used
Python

OpenCV

Streamlit

CVZone (HandTrackingModule)

Google Gemini AI (for image-based math solving)

PIL, NumPy

📌 How It Works
Raise your index finger to start drawing.

Raise your index + middle + thumb fingers to clear the canvas.

Raise only your thumb to trigger AI recognition.

The AI interprets the image and returns the solved result.
