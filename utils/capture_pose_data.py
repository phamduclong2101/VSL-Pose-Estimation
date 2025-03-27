import cv2
import mediapipe as mp
import argparse
import time
import os

from feature_extraction import *



# Temporarily ignore warning
import warnings
warnings.filterwarnings("ignore")

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

if __name__ == "__main__":
    # Get the pose name from argument
    parser = argparse.ArgumentParser("Pose Data Capture")

    # Add and parse the arguments
    parser.add_argument("--pose_name", help="Name of the pose to be save in the data folder",
                        type=str, default="La_gi")
    parser.add_argument("--confidence", help="Confidence of the model",
                        type=float, default=0.6)
    parser.add_argument("--duration", help="Duration to capture pose data",
                        type=int, default=60)
    args = parser.parse_args()

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Firstly, wait 5 seconds to get ready
    print("Get ready!")
    time.sleep(5)
    print("Capturing pose data")

    # Record the start time
    start_time = time.time()

    # Get the array
    pose_data = []

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=args.confidence,
                                      min_tracking_confidence=args.confidence)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2,
                           min_detection_confidence=args.confidence,
                           min_tracking_confidence=args.confidence)

    # Initialize drawing utility
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # Set up the holistic model
    while cap.isOpened():
        # Check if duration has passed
        if time.time() - start_time >= args.duration:
            print("End capturing")
            break

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
        pose_data.append(feature)

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

        # Convert back to BGR to render
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Display the image
        cv2.imshow('MediaPipe Face and Hand Detection', cv2.flip(image, 1))

        # Press 'q' to quit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

    # Convert to numpy array and save
    pose_data = np.array(pose_data)

    data_path = r"C:\Users\Hi Windows 11 Home\Documents\sign_recognition\Sign-Language-Classification\data_3"

    os.makedirs(data_path, exist_ok=True)
    # Save
    np.save(f"{data_path}\{args.pose_name}.npy", pose_data)
    
    print("Save pose data successfully!")



# # Thay vì mở webcam, bạn mở video từ file
# cap = cv2.VideoCapture('path_to_video.mp4')

# # Kiểm tra nếu video mở thành công
# if not cap.isOpened():
#     print("Error: Không thể mở video.")
#     exit()

# # Lặp qua từng khung hình của video
# while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#         break  # Dừng khi hết video

#     # Chuyển đổi ảnh sang định dạng RGB để MediaPipe xử lý
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Nhận diện khuôn mặt và tay
#     face_results = face_mesh.process(image)
#     hand_results = hands.process(image)

#     # Trích xuất đặc trưng
#     feature = extract_features(mp_hands, face_results, hand_results)
#     pose_data.append(feature)

#     # Vẽ các điểm mốc lên hình ảnh (có thể bỏ qua nếu không cần hiển thị)
#     if face_results.multi_face_landmarks:
#         for face_landmarks in face_results.multi_face_landmarks:
#             mp_drawing.draw_landmarks(
#                 image=image,
#                 landmark_list=face_landmarks,
#                 connections=mp_face_mesh.FACEMESH_TESSELATION,
#                 landmark_drawing_spec=None,
#                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
#             )

#     if hand_results.multi_hand_landmarks:
#         for hand_landmarks in hand_results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 image=image,
#                 landmark_list=hand_landmarks,
#                 connections=mp_hands.HAND_CONNECTIONS,
#                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
#                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
#             )

#     # Chuyển lại hình ảnh từ RGB về BGR để hiển thị (nếu cần)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     # Hiển thị video (nếu cần)
#     cv2.imshow('Pose Data Extraction', image)

#     # Nhấn 'q' để thoát
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Giải phóng tài nguyên và đóng cửa sổ
# cap.release()
# cv2.destroyAllWindows()

# # Lưu dữ liệu (sử dụng np.save như trước)
# pose_data = np.array(pose_data)
# np.save('action_data.npy', pose_data)
# print("Dữ liệu đã được lưu thành công!")
