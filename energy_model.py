import joblib
import pandas as pd

model = joblib.load("smart_energy_model.pkl")
feature_columns = joblib.load("model_features.pkl")

def predict_energy(current, temperature, month, day, hour, usage):

    voltage = 230
    peak_hour = 1 if 18 <= hour <= 22 else 0

    current_voltage_interaction = voltage * current
    temp_deviation = abs(temperature - 24)

    user_input = {
        "voltage": voltage,
        "current": current,
        "temperature_setting_C": temperature,
        "peak_hour_flag": peak_hour,
        "month": month,
        "day": day,
        "hour": hour,
        "current_voltage_interaction": current_voltage_interaction,
        "temp_deviation": temp_deviation
    }

    df = pd.DataFrame([user_input])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)

    energy_per_hour = model.predict(df)[0]

    total_energy = energy_per_hour * usage
    cost = total_energy * 7
    carbon = total_energy * 0.82

    return total_energy, cost, carbon