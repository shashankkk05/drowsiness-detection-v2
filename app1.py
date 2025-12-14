import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import av

# 1. LOAD YOUR MODEL (Cached so it runs fast)
@st.cache_resource
def load_drowsiness_model():
    # Make sure 'drowiness_new7.h5' is in the same folder
    return load_model('drowiness_new7.h5')

model = tf.keras.models.load_model("drowiness_new7.h5", compile=False)


# Load Face/Eye cascades (ensure these xml files are in your folder!)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 2. DEFINE THE PROCESSING LOGIC
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        # Convert frame from browser (AV format) to OpenCV format (numpy)
        img = frame.to_ndarray(format="bgr24")

        # --- COPY YOUR OPEN CV CODE HERE ---
        
        # 1. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Detect Faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        status = "Active" # Default status

        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Focus on eyes within the face
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # PREPARE EYE IMAGE FOR MODEL
                eye_img = roi_color[ey:ey+eh, ex:ex+ew]
                
                # Resize to what your model expects (e.g., 224x224 or 64x64)
                # CHANGE (224, 224) TO MATCH YOUR MODEL'S INPUT SHAPE!
                try:
                    final_image = cv2.resize(eye_img, (224, 224))
                    final_image = np.expand_dims(final_image, axis=0) # Add batch dimension
                    final_image = final_image / 255.0 # Normalize if you did this in training

                    # PREDICT
                    prediction = model.predict(final_image)
                    
                    # LOGIC: Adjust threshold based on your model (0.5 is standard)
                    if prediction < 0.5: # Assuming 0 = Closed/Drowsy
                        status = "Drowsy"
                except Exception as e:
                    pass

        # 3. VISUAL ALERT (Replacing Audio)
        if status == "Drowsy":
            # Flash screen RED text
            cv2.putText(img, "DROWSINESS ALERT!", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            # Draw red border
            cv2.rectangle(img, (0,0), (img.shape[1], img.shape[0]), (0,0,255), 10)
        else:
            cv2.putText(img, "Active", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Return the processed frame to the browser
        return img

# 3. BUILD THE WEBPAGE
st.title("Driver Drowsiness Detector")
st.write("Click 'Start' to activate the webcam.")

# This runs the webcam loop
webrtc_streamer(key="drowsiness-detection", video_processor_factory=VideoProcessor)