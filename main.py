import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from PIL import Image
from google import genai
import streamlit as st

client = genai.Client(api_key="AIzaSyBsGd8b4ThO02F8i8Oh8gBIRF3fBWPCzrg")


#st.set_page_config(layout="wide")
st.image("cover.png")

col1, col2 = st.columns([2,1])

with col1:
    # Create a dynamic key for the checkbox based on the layout or unique context
    run = st.checkbox('Run', value=True)  # Unique key for this checkbox
    FRAM_WINDOW = st.image([])

with col2:
    out_put = st.title("Answer")
    out_put = st.subheader("")
# Initialize the webcam to capture video
# The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
cap = cv2.VideoCapture(0)
cap.set(propId=3,value=1200)
cap.set(propId=4,value=720)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)


prev_position = None
canvas = None
img_combined = None
output_text = ""


def getHandInfo(img):
    
    # Find hands in the current frame
    # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
    # The 'flipType' parameter flips the image, making it easier for some detections
    hands, img = detector.findHands(img, draw=False, flipType=True)

    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand = hands[0]  # Get the first hand detected
        lmList = hand["lmList"]  # List of 21 landmarks for the first hand
        

        # Count the number of fingers up for the first hand
        fingers = detector.fingersUp(hand)
        print(fingers)  # Print the fingers that are up

        return fingers, lmList
    else:
        return None

def draw(info, prev_position, canvas):

    fingers, lmList = info
    current_position = None

    if fingers == [0, 1, 0, 0, 0]:  # Only index finger is up
        current_position = lmList[8][0:2]  # Index finger tip coordinates

        if prev_position is None:
            prev_position = current_position

        # Draw a line from the previous position to the current position
        cv2.line(canvas, prev_position, current_position, color=(255, 0, 255), thickness=10)
        #return current_position  # Update position for the next frame
    elif fingers == [1,0,0,0,0]:
        canvas = np.zeros_like(img)
   
    return current_position, canvas # No change if not drawing

def sendToAI(canvas, fingers):

    if fingers == [0,1,1,1,1]:
        pil_image = Image.fromarray(canvas)

        response = client.models.generate_content(
    model="gemini-2.0-flash", contents=["Solve this ", pil_image]
    )
        return response.text



# Continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)
      

    info = getHandInfo(img)

    if info:
        fingers, lmList = info
        #print(fingers)
        prev_position, canvas = draw(info, prev_position, canvas)
        output_text = sendToAI(canvas, fingers)

 
    img_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)

    FRAM_WINDOW.image(img_combined, channels="BGR")
    if output_text:
         out_put.text(output_text)
    # Display the image in a window
    # cv2.imshow("Image", img)
    #cv2.imshow("Canvas", canvas)
    #cv2.imshow("img_combined", img_combined)

    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    cv2.waitKey(1)

