#!/usr/bin/env python
# coding: utf-8

# In[8]:


import cv2
import numpy as np

# Load YOLO model and classes
net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Get output layer names (YOLO v3 has three output layers)
output_layers = net.getUnconnectedOutLayersNames()

def detect_phone_camera(frame):
    height, width, _ = frame.shape

    # Prepare the frame for YOLO input
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Set the input to the YOLO network
    net.setInput(blob)

    # Perform forward pass to get the output of the output layers
    outs = net.forward(output_layers)

    # Initialize variables to store detected object information
    class_ids = []
    confidences = []
    boxes = []

    # Analyze the output of the YOLO model
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Threshold for confidence score
                # YOLO returns center coordinates, width, and height of the bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Check if any detected object is a "cell phone" (class ID 67)
    detected_phones = []
    if len(indices) > 0 and len(boxes) > 0 and len(class_ids) > 0:
        for i in indices.flatten():
            if i < len(class_ids) and class_ids[i] == 67 and i < len(boxes):
                detected_phones.append(boxes[i])

    return detected_phones

def capture_video():
    cap = cv2.VideoCapture(0)  # 0 represents the default camera (laptop camera)

    phone_detected = False

    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)  # Use WINDOW_NORMAL to enable window resizing
    cv2.resizeWindow("Camera Feed", 800, 600)  # Set the desired window size (width, height)

    while True:
        ret, frame = cap.read()

        if ret:
            detected_phones = detect_phone_camera(frame)

            if detected_phones:
                if not phone_detected:
                    print("Phone detection started!")
                    phone_detected = True

                for box in detected_phones:
                    x, y, w, h = box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Phone", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            else:
                if phone_detected:
                    print("Phone detection stopped!")
                    phone_detected = False

            # Resize the frame to the desired size (800x600 pixels)
            resized_frame = cv2.resize(frame, (800, 600))

            cv2.imshow("Camera Feed", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Code quitting...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_video()


# In[ ]:





# In[ ]:




