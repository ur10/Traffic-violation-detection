# import torch
# import numpy as np
# from torch import nn
# import cv2 as cv
# from sklearn.model_selection import train_test_split
# import yaml
# import pandas as pd
# import ast
# import copy
# import tqdm
# import matplotlib.pyplot as plt


# class TrapRegressor(nn.Module):
#     """docstring for TrapRegressor"""

#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Linear(in_features=24, out_features=512)
#         self.laeyr2 = nn.Linear(in_features=512, out_features=256)
#         self.layer3 = nn.Linear(in_features=256, out_features=8)
#         self.tanh = nn.Tanh()

#     def forward(self, x):
#         x = self.tanh(self.layer1(x))
#         x = self.tanh(self.laeyr2(x))
#         x = self.tanh(self.layer3(x))

#         return x


# def train(hyperparam, data, model=None):
#     # Fit the model
#     torch.manual_seed(42)
#     epochs = hyperparam['epochs']

#     model = nn.Sequential(
#         nn.Linear(24, 256),
#         nn.Tanh(),
#         nn.Linear(256, 512),
#         nn.Tanh(),
#         nn.Linear(512, 7),
#     )

#     print(model)

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device)
#     # Put all data on target device TODO this on the gpu
#     X_train, y_train = torch.from_numpy(data['input_train']).type(
#         torch.float), torch.from_numpy(data['output_train']).type(torch.float)
#     X_test, y_test = torch.from_numpy(data['input_test']).type(
#         torch.float), torch.from_numpy(data['output_test']).type(torch.float)

#     X_train, y_train = X_train.to(device), y_train.to(device)
#     X_test, y_test = X_test.to(device), y_test.to(device)

#     print(len(X_train))
#     print(len(y_train))

#     loss_fn = nn.MSELoss()  # BCEWithLogitsLoss = sigmoid built-in

#     # Create an optimizer
#     optimizer = torch.optim.SGD(params=model.parameters(), lr=hyperparam['lr'])

#     batch_size = 2  # size of each batch
#     batch_start = torch.arange(0, len(X_train), batch_size)

#     # Hold the best model
#     best_mse = np.inf  # init to infinity
#     best_weights = None
#     history = []
#     train_history = []

#     # training loop
#     for epoch in range(epochs):
#         model.train()
#         with tqdm.tqdm(batch_start, unit="batch", mininterval=0,
#                        disable=True) as bar:
#             bar.set_description(f"Epoch {epoch}")
#             for start in bar:
#                 # take a batch
#                 X_batch = X_train[start:start + batch_size]
#                 y_batch = y_train[start:start + batch_size]

#                 # print(X_batch)
#                 # forward pass
#                 # print(X_batch.shape)
#                 y_pred = model(X_batch)
#                 # print(f'The model output is {y_pred}')
#                 loss = loss_fn(y_pred, y_batch)
#                 print(f'the training loss is {loss}')

#                 # backward pass
#                 optimizer.zero_grad()
#                 loss.backward()
#                 # update weights
#                 optimizer.step()
#                 # print progress
#                 loss = float(loss)
#                 train_history.append(loss)
#                 bar.set_postfix(mse=float(loss))
#         # evaluate accuracy at end of each epoch
#         model.eval()
#         y_pred = model(X_test)
#         mse = loss_fn(y_pred, y_test)
#         mse = float(mse)
#         history.append(mse)
#         # print(f'The loss is {mse}')
#         if mse < best_mse:
#             best_mse = mse
#             best_weights = copy.deepcopy(model.state_dict())

#     # restore model and return best accuracy
#     model.load_state_dict(best_weights)

#     print("MSE: %.2f" % best_mse)
#     print("RMSE: %.2f" % np.sqrt(best_mse))
#     plt.plot(train_history)
#     plt.show()

#     return model


# def create_data(file, split_size=0.2):

#     data_dict = {}
#     df = pd.read_csv('data.csv')

#     X = df['input_data'].apply(
#         lambda s: [float(x.strip(' []')) for x in s.split(',')])
#     y_label = df['output_labels'].apply(
#         lambda s: [float(x.strip(' ()')) for x in s.split(',')])
#     # print(len(y_label))
#     X = list_to_numpy(X)
#     y_label = list_to_numpy(y_label)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y_label,
#         test_size=split_size,  # 20% test, 80% train
#         random_state=42)  # make the random split reproducible
#     return X_train[0:4], X_test[0:4], y_train[0:4], y_test[0:4]


# def list_to_numpy(data):
#     # print(data[0][1])
#     array = np.zeros((len(data), len(data[0])))

#     # print(data[273])
#     for i in range(len(data)):
#         # print(i)
#         converted_array = np.array(data[i])
#         array[i] = converted_array

#     return array


# def main():
#     data = {}
#     hyperparam = {}

#     X_train, X_test, y_train, y_test = create_data("data.csv", 0.2)
#     # print((X_train[0]))
#     # print(type(y_train))
#     # X_train = list_to_numpy(X_train)
#     # X_test = list_to_numpy(X_test)
#     # y_train = list_to_numpy(y_train)
#     # y_test = list_to_numpy(y_test)

#     print(X_train.shape)
#     print(y_train.shape)

#     data['input_train'] = X_train
#     data['input_test'] = X_test
#     data['output_train'] = y_train
#     data['output_test'] = y_test

#     hyperparam['epochs'] = 2000
#     hyperparam['lr'] = 0.001

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = TrapRegressor().to(device)
#     # print(model)

#     train(hyperparam, data)


# if __name__ == "__main__":
#     main()


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yaml
import pandas as pd
import ast
import copy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tqdm

# THIS MODEL HAS BEEN CHANGED TO GIVING THE VALUES THAT ARE IN THE NORMALIZED FORMAT AND HAVE INPUT OF 48, CHANGE THE XML READER TO GIVE 24 AS INPUT WITH NORMALIZED VALUES AND THE TRAP OUTPUT SHOULD ALSO BE CHANGED TO REFLECT A TRAP INSTEAD OF A RECTANGLE.

class RegressionModel(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def create_data(file, split_size=0.3):

    data_dict = {}
    df = pd.read_csv('data.csv')

    X = df['input_data'].apply(
        lambda s: [float(x.strip(' []')) for x in s.split(',')])
    y_label = df['output_labels'].apply(
        lambda s: [float(x.strip(' ()')) for x in s.split(',')])
    # print(len(y_label))
    X = list_to_numpy(X)
    y_label = list_to_numpy(y_label)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_label,
        test_size=split_size,  # 20% test, 80% train
        random_state=42)  # make the random split reproducible
    return X_train[0:4], X_test[0:4], y_train[0:4], y_test[0:4]


# Define input, hidden, and output sizes


def list_to_numpy(data):
    # print(data[0][1])
    array = np.zeros((len(data), len(data[0])))

    # print(data[273])
    for i in range(len(data)):
        # print(i)
        converted_array = np.array(data[i])
        array[i] = converted_array

    return array



input_size = 48
hidden_size1 = 256
hidden_size2 = 512
output_size = 8

# Create an instance of the regression model
model = RegressionModel(input_size, hidden_size1, hidden_size2, output_size)

# Define a loss function (e.g., Mean Squared Error) and an optimizer (e.g., Adam)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print('creating data_dict')
X_train, X_test, y_train, y_test = create_data("data.csv", 0.2)
print('Created data')

X_train, y_train = torch.from_numpy(X_train).type(
    torch.float), torch.from_numpy(y_train).type(torch.float)
X_test, y_test = torch.from_numpy(X_test).type(
    torch.float), torch.from_numpy(y_test).type(torch.float)

# Generate example input data and target data (replace with your actual data)
input_data = X_train  # Replace with your input data
target_data = y_train  # Replace with your target data

# Training loop
num_epochs = 10000  # Adjust as needed
history = []
best_mse = np.inf
batch_size = 10  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)
for epoch in range(num_epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # take a batch
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            bar.set_postfix(mse=float(loss))

    # Print the loss at every 100 epochs (or as needed)
    if (epoch + 1) % 100 == 0:
        print(f'Epoch number is {epoch}')
        model.eval()
        y_pred = model(X_test)
        mse = criterion(y_pred, y_test)
        # mse = float(mse)
        history.append(mse.item())
        if mse.item() < best_mse:
            best_mse = mse.item()
            best_weights = copy.deepcopy(model.state_dict())

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {mse.item():.4f}')
            print(f'We get ouput as {y_pred[0]}, whereas it should have been {y_test[0]}')
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
plt.plot(history)
plt.show()