import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGLatentEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(EEGLatentEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        
        
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        
        
        
        self.flattened_size = 128 * 1 * 4
        
        
        self.fc_mean = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, x):
        
        
        # Feature Extraction Layer 1
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        # Feature Extraction Layer 2
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Feature Extraction Layer 3
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten the grid into a single 1D array per thought
        x = torch.flatten(x, start_dim=1)
        
        # Split into Mean and Variance 
        z_mean = self.fc_mean(x)
        z_logvar = self.fc_logvar(x)
        
        return z_mean, z_logvar

if __name__ == "__main__":
    
    
    fake_brainwave = torch.randn(1, 1, 14, 32) 
    
    translator = EEGLatentEncoder(latent_dim=256)
    mean, logvar = translator(fake_brainwave)
    
    print("--- Neural Translator Online ---")
    print(f"Input Biological Shape: {fake_brainwave.shape}")
    print(f"Translated Latent Space Shape: {mean.shape}")