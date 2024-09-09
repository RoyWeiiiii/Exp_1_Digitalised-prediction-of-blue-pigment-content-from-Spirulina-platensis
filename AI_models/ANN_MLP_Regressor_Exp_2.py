import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# Load data from CSV
data = pd.read_csv("D:\CodingProjects\machine_learning\Experiment 2\Combined_All_medium\Trial 1\Training\Day_Norm\Combined_Abs_Day_normalized_data_HSL.csv")

# Separate 'IndexCMYK' and 'Abs' as features (X) and 'CPC concentration' as the target variable (y)
############# With Day #############
X = data[['R','G','B','IndexRGB','Day']].values
X = data[['H','S','L','IndexHSL','Day']].values
X = data[['C','M','Y','K','IndexCMYK','Day']].values

############# With Day & Medium/Abs #############
X = data[['Abs','R','G','B','IndexRGB','Day']].values
X = data[['Abs', 'H','S','L','IndexHSL','Day']].values
X = data[['Abs', 'C','M','Y','K','IndexCMYK','Day']].values

############# Without Day #############
X = data[['R','G','B','IndexRGB']].values
X = data[['H','S','L','IndexHSL']].values
X = data[['C','M','Y','K','IndexCMYK']].values

# Y-output
y = data['CPC'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of MLPRegressor with an increased max_iter
mlp = MLPRegressor(max_iter=1000)  # Increase max_iter as needed

# Define the hyperparameter grid for hyperparameter tuning
param_grid = {
    'hidden_layer_sizes': [(32), (32, 64), (32, 64, 32), (32, 64, 64, 32), (32, 64, 128, 64, 32),(32,64,128,128,64,32),(32,64,128,256,128,64,32)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.001],
    'learning_rate': ['constant'],
}

# Create GridSearchCV object
grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, refit=True, n_jobs=-1)

# Fit grid search on training data
grid_search.fit(X_train, y_train)

# Get the best model
best_mlp = grid_search.best_estimator_

# Train the best model on the entire training dataset
best_mlp.fit(X_train, y_train)

# Make predictions on the testing dataset
y_pred = best_mlp.predict(X_test)

# Calculate and print the evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R2) Score (Accuracy): {r2}')

# Calculate Relative Error for Training and Testing Data
relative_error_train = np.abs(y_train - best_mlp.predict(X_train)) / y_train
relative_error_test = np.abs(y_test - y_pred) / y_test

# Calculate R-squared for Training and Testing Data
r2_train = r2_score(y_train, best_mlp.predict(X_train))
r2_test = r2_score(y_test, y_pred)

# Print or Display the Results
print(f'Relative Error (Training): {np.mean(relative_error_train):.4f}')
print(f'Relative Error (Testing): {np.mean(relative_error_test):.4f}')
print(f'R-squared (Training): {r2_train:.4f}')
print(f'R-squared (Testing): {r2_test:.4f}')

# Capture and print the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:")
print(best_params)

# Create a scatter plot to visualize actual vs. predicted CPC concentrations
plt.figure(figsize=(10, 6))
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2, label='Perfect Prediction', alpha=0.7)
plt.scatter(y_test, y_test, color='red', alpha=0.7, label='Actual DCW Concentration', marker='o')
plt.scatter(y_test, y_pred, color='blue', alpha=0.7, label='Predicted DCW Concentration', marker='x')
plt.xlabel('Actual DCW Concentration')
plt.ylabel('Predicted DCW Concentration')
plt.title('Actual vs. Predicted CPC Concentration')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Display the number of iterations and batch size used
num_iterations = best_mlp.n_iter_
batch_size = best_mlp.batch_size
print(f'Number of Iterations: {num_iterations}')
print(f'Batch Size: {batch_size}')