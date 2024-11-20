#Import Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#########################################################
#Define the Restyled U-Net Model
class MultiDilationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates):
        super(MultiDilationBlock, self).__init__()
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate)
            for rate in dilation_rates
        ])
        self.batch_norm = nn.BatchNorm2d(out_channels * len(dilation_rates))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = torch.cat([conv(x) for conv in self.dilated_convs], dim=1)
        out = self.batch_norm(out)
        return self.relu(out)

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    
    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        value = self.value(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        out = torch.bmm(attention, value).permute(0, 2, 1).view(batch_size, C, H, W)
        return out

class RestyledUNet(nn.Module):
    def __init__(self):
        super(RestyledUNet, self).__init__()
        self.enc1 = MultiDilationBlock(1, 64, dilation_rates=[1, 2])
        self.pool = nn.MaxPool2d(2)
        self.enc2 = MultiDilationBlock(64 * 2, 128, dilation_rates=[1, 2])
        self.attention = AttentionModule(128 * 2)
        self.bottleneck = nn.Conv2d(128 * 2, 256, kernel_size=3, padding=1)
        self.up1 = nn.ConvTranspose2d(256, 128 * 2, kernel_size=2, stride=2)
        self.dec1 = nn.Conv2d(128 * 4, 64, kernel_size=3, padding=1)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x3 = self.enc2(x2)
        x4 = self.pool(x3)
        x5 = self.attention(x4)
        x5 = self.bottleneck(x5)
        x6 = self.up1(x5)
        x7 = torch.cat([x6, x3], dim=1)
        x8 = self.dec1(x7)
        return torch.sigmoid(self.final(x8))
################################################################################################
#Define Loss Function
def geometric_loss(predictions, targets, lambda_c=0.01, lambda_p=0.01, epsilon=0.01):
    bce_loss = nn.BCELoss()(predictions, targets)
    grad_x = torch.abs(predictions[:, :, 1:, :] - predictions[:, :, :-1, :])
    grad_y = torch.abs(predictions[:, :, :, 1:] - predictions[:, :, :, :-1])
    curvature_penalty = torch.mean(grad_x + grad_y)
    return bce_loss + lambda_c * curvature_penalty
#################################################################################################
#Load Image and Preprocess
def load_image(image_path):
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

image_path = r"C:\Users\Guman.Garayev\Desktop\Wrinkles_1.jpg"
image_tensor = load_image(image_path)
#################################################################################################
#Simulate Model on Image
# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RestyledUNet().to(device)
model.eval()

# Run the model on the input image
image_tensor = image_tensor.to(device)
with torch.no_grad():
    predictions = model(image_tensor)
    predicted_mask = predictions.squeeze().cpu().numpy()

# Visualize the results
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_tensor.cpu().squeeze(), cmap="gray")
plt.subplot(1, 2, 2)
plt.title("Predicted Wrinkle Map")
plt.imshow(predicted_mask, cmap="jet")
plt.show()
