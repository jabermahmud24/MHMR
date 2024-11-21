import logging
import pprint
import time
import cv2
from digit_interface.digit import Digit
from digit_interface.digit_handler import DigitHandler
logging.basicConfig(level=logging.DEBUG)

# Connect to a Digit device with serial number with friendly name
digit1 = Digit("D20895", "Index")
digit1.connect()

# Change LED illumination intensity
digit1.set_intensity(Digit.LIGHTING_MIN)
time.sleep(1)
digit1.set_intensity(Digit.LIGHTING_MAX)

# Change DIGIT resolution to QVGA
qvga_res = Digit.STREAMS["QVGA"] #lower resolution 320x240
digit1.set_resolution(qvga_res)

# Change DIGIT FPS to 15fps
fps_30 = Digit.STREAMS["QVGA"]["fps"]["30fps"]
digit1.set_fps(fps_30)

# Grab single frame from DIGIT
frame = digit1.get_frame()
# print(f"Frame WxH: {frame.shape[0]}{frame.shape[1]}")

# Display stream obtained from DIGIT
digit1.show_view()

# Disconnect DIGIT stream
digit1.disconnect()


# Find a Digit by serial number and connect manually with opencv
digit = DigitHandler.find_digit("D20895")
pprint.pprint(digit)
cap = cv2.VideoCapture(digit["dev_name"])
if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    while True:
        # Capture a single frame
        ret, frame = cap.read()

        # Check if frame is captured successfully
        if not ret:
            print("Failed to capture frame")
            break

        # Display the frame
        cv2.imshow("DIGIT Feed", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()