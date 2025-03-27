import cv2
import mediapipe as mp
import argparse
import torch

from utils.train import LSTMModel
from utils.feature_extraction import *
from utils.strings import *
from utils.model import ASLClassificationModel
from config import MODEL_NAME, MODEL_CONFIDENCE

import streamlit as st

# Temporarily ignore warning
import warnings
warnings.filterwarnings("ignore")

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

if __name__ == "__main__":
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Create message handler
    expression_handler = ExpressionHandler()

    # Streamlit app
    st.set_page_config("VIETNAM Sign Language", page_icon="magnifying glass", layout="wide")
    st.markdown('<h1 style="text-align: center;">🦾 Sign language recognition system</h1>', unsafe_allow_html=True)
    st.header("Put your face in the frame")

    # Initialize session state for History
    if 'history' not in st.session_state:
        st.session_state.history = []  # Initialize history as an empty list

    with st.sidebar:
        st.subheader("History")
        # Show the history in the sidebar
        for message in st.session_state.history:
            st.markdown(f"- {message}")

    # Create two columns
    col1, col2 = st.columns([4, 2])

    # Create a placeholder for the webcam feed in the first column
    with col1:
        video_placeholder = st.empty()

    # Create a placeholder for prediction text in the second column
    with col2:
        prediction_placeholder = st.empty()

    # Load model
    print("Initialising model ...")
    model_path = r"C:\Users\Hi Windows 11 Home\Documents\sign_recognition\Sign-Language-Classification\models"
    model = LSTMModel(input_size=86, hidden_size=64, num_classes=12)
    model.load_state_dict(torch.load(f"{model_path}/{MODEL_NAME}"))

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=MODEL_CONFIDENCE,
                                      min_tracking_confidence=MODEL_CONFIDENCE)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2,
                           min_detection_confidence=MODEL_CONFIDENCE,
                           min_tracking_confidence=MODEL_CONFIDENCE)

    # Initialize drawing utility
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # Starting the application
    print("Starting application")

    mapping = {0: "Ban", 1: '',2: "Cam on", 3: 'F',4: "La gi ?", 5: "P", 6: 'T',7: "Tam biet", 8: "Ten", 9: "Toi", 10: 'Xin chao', 11: 'Yeu'}


    # Set up the holistic model
    while cap.isOpened():
        # Check if getting frame is successful
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the image to RGB
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and find faces
        face_results = face_mesh.process(image)

        # Process the image and find hands
        hand_results = hands.process(image)

        # Extract feature from face and hand results
        feature = extract_features(mp_hands, face_results, hand_results)

        # Convert feature to tensor and make prediction
        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)  # Unsqueeze to add batch dimension
        with torch.no_grad():  # Disable gradient calculation during inference
            output = model(feature_tensor)  # Forward pass through the model
            _, predicted = torch.max(output, 1)  # Get the predicted class

        # Convert predict to text
        # expression = model.predict(feature)
        predicted_label = predicted.item()

        # expression = model.predict(feature)
        expression_handler.receive(predicted_label)

        # Add the current expression to history
        # st.session_state.history.append(expression_handler.get_message())

        # Draw the face mesh annotations on the image
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )

        # Draw the hand annotations on the image
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

        # Display the image and prediction
        video_placeholder.image(image, channels="RGB", use_column_width=True)
        prediction_placeholder.markdown(f'''<h2 class="big-font">{expression_handler.get_message()}</h2>''', unsafe_allow_html=True)

        # Press 'q' to quit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
