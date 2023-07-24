# basics
import numpy as np
import os
import io
import time
# computer vision
import cv2
from paddleocr import PaddleOCR
# audio
from gtts import gTTS
# local modules
from utils import select, draw_viz, check_rotation

# Parameters
# test = "real-time"
test  = "equipment_label"

if test == "real-time": source = 0
elif test == "equipment_label": source = "./Test Videos/Equipment Label.MOV"

# Initialize the camera
cap = cv2.VideoCapture(source)
# rotateCode = check_rotation(source)
# Initialize OCR
ocr = PaddleOCR(lang="en", show_log = False)



def main():
    # spoken = ''
    speaking = ''
    spoken = set()
    pause_until = time.time()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame not grabbed -> end of the video
        if not ret:
            break

        # # Rotate the image
        # if rotateCode is not None:
        #     frame = cv2.rotate(frame, rotateCode)

        # Convert the image from OpenCV BGR format to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect text
        texts = ocr.ocr(image, cls=False)[0]

        # Select text
        # target = tuple(np.array(image.shape[:2]) // 2) 
        target = image.shape[1]//2, image.shape[0]//2
        selection = select(image, texts=texts, target=target)
        frame = draw_viz(image, texts=texts, selection=selection, cursor=target)

        # Display the resulting frame
        cv2.imshow('Camera Stream', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Produce speech
        text = selection['name'][0] if selection else ''
        # if text != spoken and text != '':
        if (time.time() > pause_until) and (text != '') and (text not in spoken):
            # print(text, ':', spoken) # DEBUG
            print(text, ':', speaking) # DEBUG
            tts = gTTS(text=text, lang='en')

            tts.save("hello.mp3")

            os.system("start hello.mp3")
            # spoken = text
            speaking = text
            spoken.add(text)

            # time.sleep(1)
            pause_until = time.time() + 1

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": main()