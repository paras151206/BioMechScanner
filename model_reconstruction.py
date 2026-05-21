import torch
import torch.nn as nn


class BrainToImageDecoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(BrainToImageDecoder, self).__init__()
        
        
        self.fc = nn.Linear(latent_dim, 128 * 16 * 16)
        
        
        self.decoder = nn.Sequential(
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh() 
            
            
        )

    def forward(self, z):
        
        x = self.fc(z)
        x = x.view(-1, 128, 16, 16) 
        img = self.decoder(x)
        return img


class ImageDiscriminator(nn.Module):
    def __init__(self):
        super(ImageDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 1),
            nn.Sigmoid() 
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


if __name__ == "__main__":

    fake_latent_vector = torch.randn(1, 256) 
    
    decoder = BrainToImageDecoder(latent_dim=256)
    discriminator = ImageDiscriminator()
    
    # Pass the thought into the Decoder
    generated_image = decoder(fake_latent_vector)
    
    # Pass the image into the Discriminator
    judgment = discriminator(generated_image)
    
    print("--- Reconstruction Engine Online ---")
    print(f"Input Latent Thought: {fake_latent_vector.shape}")
    print(f"Generated Image Shape: {generated_image.shape}")
    print(f"Discriminator Judgment: {judgment.shape} (Score: {judgment.item():.4f})")