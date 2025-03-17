from flask import Flask, render_template, request
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Define the SiameseNetwork class
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = SiameseNetwork()
model.load_state_dict(torch.load('snn_modell.pth'))
model.eval()

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define preprocessing for input images
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

def preprocess_image(image):
    image = Image.open(image).convert("L")  # Convert to grayscale
    image = transform(image)
    return image.unsqueeze(0).to(device)  # Add batch dimension and move to device

def predict(image1, image2):
    img1 = preprocess_image(image1)
    img2 = preprocess_image(image2)

    with torch.no_grad():
        output1, output2 = model(img1, img2)  # Model should be on the same device
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
    
    return euclidean_distance.item()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file1 = request.files['image1']
        file2 = request.files['image2']

        distance = predict(file1, file2)
        threshold = 0.0015
        similarity = distance < threshold

        return render_template('index.html', distance=distance, similarity=similarity)

    return render_template('index.html', distance=None)

if __name__ == '__main__':
    app.run(debug=True)
