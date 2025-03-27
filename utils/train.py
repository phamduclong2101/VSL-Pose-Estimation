# import os
# import numpy as np
# import argparse
# import pickle

# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split

# if __name__ == "__main__":
#     # Get the pose name from argument
#     parser = argparse.ArgumentParser("Training model")

#     # Add and parse the arguments
#     parser.add_argument("--model_name", help="Name of the model",
#                         type=str, default="model")
#     parser.add_argument("--dir", help="Location of the model",
#                         type=str, default="models")
#     args = parser.parse_args()

#     # Train X, y and mapping
#     X, y, mapping = [], [], dict()

#     # Read in the data from data folder
#     for current_class_index, pose_file in enumerate(os.scandir("data_1")):
#         # Load pose data
#         file_path = f"data_1/{pose_file.name}"
#         pose_data = np.load(file_path)

#         # Add to training data
#         X.append(pose_data)
#         y += [current_class_index] * pose_data.shape[0]

#         # Add to mapping
#         mapping[current_class_index] = pose_file.name.split(".")[0]

#     # Convert to Numpy
#     X, y = np.vstack(X), np.array(y)

#     # Create model
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#     # Train the model and validate
#     model = SVC(decision_function_shape='ovo', kernel='rbf')
#     model.fit(X_train, y_train)

#     # Get the train and test accuracy
#     train_accuracy, test_accuracy = model.score(X_train, y_train), model.score(X_test, y_test)

#     # Display the train and test accuracy
#     print(f"Training examples: {X.shape[0]}. Num classes: {len(mapping)}")
#     print(f"Train accuracy: {round(train_accuracy * 100, 2)}% - Test accuracy: {round(test_accuracy * 100, 2)}%")

#     # Save the model to the model's folder
#     model_path = os.path.join(f"{args.dir}", f"{args.model_name}.pkl")
#     with open(model_path, "wb") as file:
#         pickle.dump((model, mapping), file)
#     print(f"Saved model to {model_path}")


import os
import numpy as np
import argparse
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Định nghĩa mô hình LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Sử dụng đầu ra của LSTM tại bước cuối cùng

        if len(lstm_out.shape) == 3:
            out = self.fc(self.dropout(lstm_out[:, -1, :]))  # LSTM output at last time step
        else:
            out = self.fc(self.dropout(lstm_out))
        return out

if __name__ == "__main__":
    # Get the pose name from argument
    parser = argparse.ArgumentParser("Training model")

    # Add and parse the arguments
    parser.add_argument("--model_name", help="Name of the model", type=str, default="model")
    parser.add_argument("--dir", help="Location of the model", type=str, default="models")
    args = parser.parse_args()

    # Train X, y and mapping
    X, y, mapping = [], [], dict()

    # Read in the data from data folder
    for current_class_index, pose_file in enumerate(os.scandir("data_3")):
        # Load pose data
        file_path = f"data_3/{pose_file.name}"
        pose_data = np.load(file_path)

        # Add to training data
        X.append(pose_data)
        y += [current_class_index] * pose_data.shape[0]

        # Add to mapping
        mapping[current_class_index] = pose_file.name.split(".")[0]

    # Convert to Numpy
    X, y = np.vstack(X), np.array(y)

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Chuyển đổi dữ liệu thành dạng PyTorch Tensor và chuẩn hóa
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Tạo DataLoader cho huấn luyện và kiểm tra
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Khởi tạo mô hình
    input_size = X_train.shape[1]  # Số lượng đặc trưng cho mỗi khung hình
    hidden_size = 64  # Kích thước của lớp ẩn LSTM
    num_classes = len(mapping)  # Số lớp hành động
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

    # Sử dụng Adam optimizer và CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Huấn luyện mô hình
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()  # Đặt mô hình ở chế độ huấn luyện
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Tính toán độ chính xác
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        train_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Train Accuracy: {train_accuracy}%")

    # Đánh giá mô hình trên tập kiểm tra
    model.eval()  # Đặt mô hình ở chế độ đánh giá
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy}%")


    # Display the train and test accuracy
    print(f"Training examples: {X.shape[0]}. Num classes: {len(mapping)}")
    print(f"Train accuracy: {round(train_accuracy * 100, 2)}% - Test accuracy: {round(test_accuracy * 100, 2)}%")


    # Lưu mô hình
    model_path = os.path.join(f"{args.dir}", f"{args.model_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    