import cv2
import mediapipe as mp
import argparse
import torch
import json  # Thêm import này

from train import LSTMModel
from feature_extraction import *
from strings import *
from model import ASLClassificationModel
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

    from config import MODEL_NAME, MODEL_CONFIDENCE
    print("Initialising model ...")
    model_path = r"C:\Users\Hi Windows 11 Home\Documents\sign_recognition\Sign-Language-Classification\models"
    model_file = f"{model_path}/{MODEL_NAME}"

    # Sử dụng GPU nếu có
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LSTMModel(input_size=86, hidden_size=64, num_classes=12).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))

    # Load mapping từ file json (tự động theo model)
    mapping_path = model_file.replace(".pth", "_mapping.json")
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    mapping = {int(k): v for k, v in mapping.items()}

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

        # Convert feature to tensor and make prediction (chuyển tensor sang device)
        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(feature_tensor)
            _, predicted = torch.max(output, 1)

        # Convert predict to text
        predicted_label = predicted.item()
        predicted_text = mapping.get(predicted_label, "")

        # Gửi kết quả cho handler
        expression_handler.receive(predicted_label)

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
        prediction_placeholder.markdown(f'''<h2 class="big-font">{predicted_text}</h2>''', unsafe_allow_html=True)

        # Press 'q' to quit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
