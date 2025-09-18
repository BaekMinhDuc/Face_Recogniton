from ultralytics import YOLO

# Load the YOLO model
model = YOLO("face_detection.pt")  # Load the pretrained YOLOv8n-face model

# Run predictions on the webcam (source=0)
model.predict(source=0, show=True)  # Use the webcam as input and display results