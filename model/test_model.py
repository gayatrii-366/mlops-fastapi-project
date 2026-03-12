import pickle
import numpy as np

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(sample)

print("Prediction:", prediction)