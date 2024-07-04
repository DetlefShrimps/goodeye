    Dense(64, activation='relu'),
    Dense(len(target_columns), activation='linear')
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
y_pred_nn = model

# Predict using the neural network model
y_pred_nn = model.predict(X_test)

# Evaluate the neural network model
nn_mse = mean_squared_error(y_test, y_pred_nn)
print(f"Neural Network MSE: {nn_mse}")

# Save the models and scaler for future use
os.makedirs('models', exist_ok=True)
joblib.dump(rf, 'models/random_forest_regressor.pkl')
joblib.dump(gb, 'models/gradient_boosting_regressor.pkl')
model.save('models/neural_network_model.h5')
joblib.dump(scaler, 'models/scaler.pkl')

print("Models and scaler saved successfully.")

# Function to load models and scaler
def load_models():
    rf = joblib.load('models/random_forest_regressor.pkl')
    gb = joblib.load('models/gradient_boosting_regressor.pkl')
    model = tf.keras.models.load_model('models/neural_network_model.h5')
    scaler = joblib.load('models/scaler.pkl')
    return rf, gb, model, scaler

# Function to make predictions using the loaded models
def make_predictions(new_data):
    rf, gb, model, scaler = load_models()
    new_data_scaled = scaler.transform(new_data)
    rf_pred = rf.predict(new_data_scaled)
    gb_pred = gb.predict(new_data_scaled)
    nn_pred = model.predict(new_data_scaled)
    return rf_pred, gb_pred, nn_pred

# Example usage of prediction function
# Assuming new_data is a DataFrame with the same structure as the training data
# new_data = pd.DataFrame(...)  # Replace with actual new data
# rf_pred, gb_pred, nn_pred = make_predictions(new_data)
# print("Random Forest predictions:", rf_pred)
# print("Gradient Boosting predictions:", gb_pred)
# print("Neural Network predictions:", nn_pred)
