import pickle
import numpy as np

# Generate completely random numbers matching the exact shape your app expects
fake_payload = {
    'x_test': np.random.randn(10, 14, 32, 1),
    'y_test': np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]) # Hardcodes class index 2 (Dog!)
}

with open("fake_data.pkl", "wb") as f:
    pickle.dump(fake_payload, f)

print("Created 'fake_data.pkl' successfully!")