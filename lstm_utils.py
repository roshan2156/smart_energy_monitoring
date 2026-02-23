import numpy as np
import joblib
from tensorflow.keras.models import load_model

lstm_model = load_model("lstm_energy_model.h5")
scaler = joblib.load("lstm_scaler.pkl")

sequence_length = 7

def forecast_energy(history, days):

    predictions = []
    current_sequence = history.copy()

    for _ in range(days):
        pred = lstm_model.predict(
            current_sequence.reshape(1, sequence_length, 1),
            verbose=0
        )[0][0]

        predictions.append(pred)
        current_sequence = np.vstack((current_sequence[1:], [[pred]]))

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1,1)
    )

    return predictions.flatten()