import pickle
import numpy as np


file_path = r"C:\Projects\Dream_Decoder\data\eeg_tensors\eeg\char\data.pkl"

print("Bypassing legacy encoding... Cracking open the Pickle file...")

try:
    with open(file_path, 'rb') as f:
        
        data = pickle.load(f, encoding='latin1')
        
    print(f"\n--- Decryption Successful ---")
    print(f"Primary Data Type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Found Keys: {list(data.keys())}")
        print("\n--- Detailed Breakdown ---")
        for key, value in data.items():
            print(f"Key: '{key}' | Type: {type(value)}")
            
            if hasattr(value, 'shape'):
                print(f"  -> Shape: {value.shape}")
            elif isinstance(value, list):
                print(f"  -> Length: {len(value)}")
                
    elif hasattr(data, 'shape'):
        print(f"Data Shape: {data.shape}")
        
    else:
        print("Data is a standard Python object. Preview:")
        print(data)

except Exception as e:
    print(f"Failed to open pickle: {e}")