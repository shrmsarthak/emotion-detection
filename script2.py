import cv2
import numpy as np

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'rgb', "size": (640, 480)}))
picam2.start()

# Start OpenCV window thread
cv2.startWindowThread()

def detect_emotion():
    # Load the TensorFlow Lite model
    interpreter = cv2.dnn.readNetFromTensorflow('emotion_recognition.tflite')

    # Define the emotion labels
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    while True:
        # Capture image from Picamera2
        im = picam2.capture_array()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.astype("float32") / 255.0

            # Preprocess input data (similar to what you did for Keras)
            face_roi = np.expand_dims(face_roi, axis=0)

            # Set input tensor
            interpreter.setTensor(face_roi)

            # Run inference
            interpreter.invoke()

            # Get the output tensor
            output_data = interpreter.getTensor()

            # Get the predicted class label
            label = emotion_labels[np.argmax(output_data)]

            # Display the emotion label on the frame
            cv2.putText(im, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Emotion Detection', im)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Stop Picamera2
    picam2.stop()
    # Close OpenCV windows
    cv2.destroyAllWindows()

# Run the emotion detection function
detect_emotion()
