import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. Define the NN-3 Model
def create_nn_model(input_dim):
    model = Sequential([
        Dense(32, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(18, activation='relu'),
        Dense(1)  # Predicts risk premium
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 2. Train the Model
X_train, y_train = load_training_data()  # Replace with your data loading
model = create_nn_model(input_dim=X_train.shape[1])
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2)

# 3. Generate Predictions with Dropout
def predict_with_uncertainty(f_model, X, n_simulations=100):
    predictions = np.array([f_model(X, training=True).numpy().flatten() for _ in range(n_simulations)])
    mean_prediction = predictions.mean(axis=0)
    variance_prediction = predictions.var(axis=0)
    return mean_prediction, variance_prediction

X_test = load_test_data()  # Replace with your test data
mean_preds, variance_preds = predict_with_uncertainty(model, X_test)

# 4. Compute Confidence Intervals
z_score = 1.96  # 95% confidence interval
lower_bound = mean_preds - z_score * np.sqrt(variance_preds)
upper_bound = mean_preds + z_score * np.sqrt(variance_preds)

# 5. Results
print("Predicted Risk Premiums:", mean_preds)
print("Confidence Intervals:", list(zip(lower_bound, upper_bound)))
