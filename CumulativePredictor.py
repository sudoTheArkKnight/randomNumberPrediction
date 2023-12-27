import numpy as np
from sklearn.linear_model import LinearRegression
import random

# Initialize the model
model = LinearRegression()

# Function to update the model with accumulated data
def update_model(model, accumulated_data, new_data):
    X_accumulated = accumulated_data[:-1].reshape(-1, 1)
    y_accumulated = accumulated_data[1:]
    
    # Concatenate old and new data
    X_combined = np.concatenate((X_accumulated, new_data[:-1].reshape(-1, 1)))
    y_combined = np.concatenate((y_accumulated, new_data[1:]))
    
    # Update the model with combined data
    model.fit(X_combined, y_combined)
    
    # Reshape the scalar to have one dimension
    updated_input = np.array([X_combined[-1][-1]])
    
    # Return the combined data for future updates
    return updated_input, y_combined[-1]

# Train the initial model with the first dataset
accumulated_data = np.random.randint(0, 100, 1000)
update_model(model, accumulated_data, accumulated_data)

# Generate and update the model with new datasets
for i in range(5):  # Update with 5 new datasets
    new_data = np.random.randint(0, 100, 1000)
    
    # Display the actual and predicted next numbers before the update
    next_random_number = random.randint(0, 100)
    next_number_prediction = model.predict([[new_data[-1]]])
    print(f'Before Update - Actual: {next_random_number}, Predicted: {next_number_prediction[0]}')
    
    # Update the model with new data and accumulated data
    updated_input, updated_output = update_model(model, accumulated_data, new_data)
    
    # Display the actual and predicted next numbers after the update
    next_random_number = random.randint(0, 100)
    next_number_prediction = model.predict([updated_input])
    print(f'After Update - Actual: {next_random_number}, Predicted: {next_number_prediction[0]}')
    print('---')

# You can continue this loop with new datasets to further improve the model
