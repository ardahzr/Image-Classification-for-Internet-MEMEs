import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np

# Load CSV file
df = pd.read_csv('/content/archive/MIMIC2024.csv')

# Augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Extract labels directly from the DataFrame
labels = df[['Misogyny', 'Objectification', 'Prejudice', 'Humiliation']].values

# Define directories
train_directory = '/content/archive/train'
val_directory = '/content/archive/val'
test_directory = '/content/archive/test'


class CustomDataset(Dataset):
    def __init__(self, df, labels, directory, transform=None):
        self.df = df
        self.labels = labels
        self.directory = directory
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        img_path = f"{self.directory}/{img_name}"

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            return self.__getitem__((idx + 1) % len(self.df))

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


# Create datasets and loaders
train_dataset = CustomDataset(df, labels, train_directory, transform=transform)
val_dataset = CustomDataset(df, labels, val_directory, transform=transform)
test_dataset = CustomDataset(df, labels, test_directory, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


model = models.resnet152(pretrained=True)
num_features = model.fc.in_features


model.fc = nn.Sequential(
    nn.Linear(num_features, 64),
    nn.ReLU(),
    nn.Dropout(0.6),
    nn.Linear(64, 4),
    nn.Sigmoid()  
)

# GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Training 
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        print(f'Epoch {epoch + 1}/{num_epochs} - Training Loss: {epoch_loss:.4f}')

    print('Training complete')

# Train the model
train_model(model, criterion, optimizer, train_loader, val_loader)


torch.save(model.state_dict(), 'model.pth')


model.load_state_dict(torch.load('model.pth'))
model.eval()

test_loss = 0.0

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Evaluating"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)

test_loss /= len(test_loader.dataset)
print(f'Test Loss: {test_loss:.4f}')

# Make predictions on test data
test_predictions = []
file_names = []

with torch.no_grad():
    for inputs, _ in tqdm(test_loader, desc="Predicting"):
        inputs = inputs.to(device)
        outputs = model(inputs)
        test_predictions.append(outputs.cpu().numpy())
        file_names.extend([test_dataset.df.iloc[idx, 0] for idx in range(len(inputs))])

# Convert list of predictions to numpy array
test_predictions = np.vstack(test_predictions)

pred_df = pd.DataFrame(test_predictions, columns=['Misogyny', 'Objectification', 'Prejudice', 'Humiliation'])
pred_df['Filename'] = file_names

pred_df = pred_df[['Filename', 'Misogyny', 'Objectification', 'Prejudice', 'Humiliation']]

# Save 
pred_df.to_csv('/content/archive/test_predictions.csv', index=False)