import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Generate a random sequence of 1000 numbers
np.random.seed(42)
random_sequence = np.random.randint(0, 100, 1000)

# Step 2: Data Preprocessing
X = random_sequence[:-1].reshape(-1, 1)
y = random_sequence[1:]

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Choose and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Model Evaluation
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy}')

# Step 6: Prediction
next_number_prediction = model.predict([[random_sequence[-1]]])
print(f'Predicted Next Number: {next_number_prediction[0]}')
