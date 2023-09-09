import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

# Load the trained model
loaded_model = tf.keras.models.load_model('trained_model.h5')

# List of class labels
class_labels = ['class_1', 'class_2', 'class_3', 'class_4', 'class_5']


# Function to predict the class of an image
def predict_image_class(img):
    img = cv2.resize(img, (64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    prediction = loaded_model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100
    class_label = class_labels[predicted_class]
    return class_label, confidence


# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 for the default camera

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Predict the class of the captured frame
    predicted_class, confidence = predict_image_class(frame)

    # Display the result on the frame
    cv2.putText(frame, f"Class: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()
