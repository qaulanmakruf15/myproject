import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# mengulang setiap kalimat dalam intents pattern
for intent in intents['intents']:
    tag = intent['tag']
    #menambahkan tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        #tokenize setiap kata dalam kalimat
        w = tokenize(pattern)
        #tambahkan ke daftar kata 
        all_words.extend(w)
        #tambahkan ke xy pair
        xy.append((w, tag))

# stem and lower setiap kata
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# menghapus duplicate dan sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

#membuat training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    #bag of words untuk setiap pattern sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss hanya membutuhkan class labels, bukan one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 500
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    # support indexing sedemikian rupa sehingga kumpulan data [i] dapat digunakan untuk mendapatkan sampel ke-i
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    # bisa memanggil len(dataset) untuk return size nya
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

predicted_test = []
labels_l = []
actual_values = []
predicted_values = []

N = len(train_loader)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(words)
        predicted = outputs.data.max(1)[1]
        predicted_test.append(predicted.cpu().numpy())
        labels_l.append(labels.cpu().numpy())
        # jika y akan menjadi one-hot, harus aplikasikan
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
         # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predicted_values.append(np.concatenate(predicted_test).ravel())
    actual_values.append(np.concatenate(labels_l).ravel())
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        print('training accuracy : {:.2f} %'.format(100 * len((np.where(np.array(predicted_values[0])==(np.array(actual_values[0])))[0])) / len(actual_values[0])))
        
print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
