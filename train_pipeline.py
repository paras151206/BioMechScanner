import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from model_encoder import EEGLatentEncoder
from model_reconstruction import BrainToImageDecoder

def train_system():
    print("=== Initializing Optimization Training Engine ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing calculations on target device: {device}\n")

    
    data_path = r"C:\Projects\Dream_Decoder\data\eeg_tensors\eeg\char\data.pkl"
    print("Loading biological tensors into memory...")
    with open(data_path, 'rb') as f:
        payload = pickle.load(f, encoding='latin1')
    
    
    x_train = payload['x_train'] 
    
    
    x_train_tensor = torch.from_numpy(x_train).permute(0, 3, 1, 2).float()
    
    print(f"Training dataset loaded: {x_train_tensor.shape[0]} unique thought vectors.")

    
    dataset = TensorDataset(x_train_tensor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    
    encoder = EEGLatentEncoder(latent_dim=256).to(device)
    decoder = BrainToImageDecoder(latent_dim=256).to(device)
    
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    criterion = nn.MSELoss()

    encoder.train()
    decoder.train()


    epochs = 3
    print("\n=== Beginning Network Weight Optimization ===")
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (brainwaves,) in enumerate(dataloader):
            brainwaves = brainwaves.to(device)
            
            
            optimizer.zero_grad()
            
            
            mean, logvar = encoder(brainwaves)
            reconstructed_output = decoder(mean)
            
            

            loss = criterion(reconstructed_output, torch.zeros_like(reconstructed_output).to(device))
            
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(dataloader)}] | Current Loss: {loss.item():.6f}")

    print("\nOptimization sequence complete!")
    
    
    print("Saving trained network matrices...")
    torch.save(encoder.state_dict(), "encoder_weights.pth")
    torch.save(decoder.state_dict(), "decoder_weights.pth")
    print("Weights successfully secured: 'encoder_weights.pth' and 'decoder_weights.pth' are live!")

if __name__ == "__main__":
    train_system()