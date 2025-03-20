import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import pywt
from matplotlib import pyplot as plt
from models import TimeSeriesModel

# Load and clean the datase

import pandas as pd

wind_4_df = pd.read_csv('wind_4_seconds_dataset.tsf', sep='/t')
wind_4_df = list(map(float,wind_4_df.iloc[14]['# Dataset Information'].split(':')[2].split(',')))
time_series = np.array(wind_4_df)
# Normalize the time series data using Min-Max scaling
scaler = MinMaxScaler()
time_series_scaled = scaler.fit_transform(time_series.reshape(-1, 1)).flatten()

# Define context size
context_size = 20

def wavelet_transform(data, wavelet, level=1):
    coeffs = pywt.swt(data, wavelet, level=level)
    details = coeffs[0][1]
    # Normalize the wavelet coefficients to have similar scale as the original data
    details = (details - np.mean(details)) / (np.std(details) + 1e-8) * np.std(data)
    return details

# Apply wavelet transform to sequences
def apply_wavelet_to_sequences(sequences, wavelet, level=1):
    """Apply wavelet transform to each sequence in the dataset"""
    wavelet_sequences = []
    for seq in sequences:
        wavelet_features = wavelet_transform(seq, wavelet, level)

        # Combine original sequence with its wavelet features
        combined = np.column_stack((seq, wavelet_features))
        wavelet_sequences.append(combined)
    return np.array(wavelet_sequences)

# Apply multiple wavelet transforms to sequences
def apply_multiple_wavelets_to_sequences(sequences, wavelets, level=1):
    """Apply multiple wavelet transforms to each sequence in the dataset"""
    wavelet_sequences = []
    for seq in sequences:
        # Start with the original sequence
        features = [seq]
        # Add features from each wavelet transform
        for wavelet in wavelets:
            wavelet_features = wavelet_transform(seq, wavelet, level)
            features.append(wavelet_features)
        # Combine all features
        combined = np.column_stack(features)
        wavelet_sequences.append(combined)
    return np.array(wavelet_sequences)

def train_evaluate_multiple_wavelets(time_series_scaled, wavelets, context_size):
    print(f"\n\n===== Training model with wavelets: {', '.join(wavelets)} =====\n")

    # Create sequences from the original time series
    X, y = create_sequences(time_series_scaled, context_size)

    # Apply multiple wavelet transforms to sequences
    X_wavelet = apply_multiple_wavelets_to_sequences(X, wavelets)

    split_index = len(X_wavelet) - context_size

    # Split the data
    X_train, X_val = X_wavelet[:split_index], X_wavelet[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Create DataLoaders
    batch_size = 32
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define the model - input size is number of wavelets + 1 (original sequence)
    input_size = len(wavelets) + 1
    hidden_size = 128
    num_layers = 2
    model = TimeSeriesModel(input_size, hidden_size, num_layers)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50

    best_val_mse = float('inf')
    best_model_state = None

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * X_batch.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # Evaluate on validation set
        model.eval()
        running_val_mse = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_outputs = model(X_batch)
                val_loss = criterion(val_outputs, y_batch)
                running_val_mse += val_loss.item() * X_batch.size(0)

        epoch_val_mse = running_val_mse / len(val_loader.dataset)

        if epoch_val_mse < best_val_mse:
            best_val_mse = epoch_val_mse
            best_model_state = model.state_dict()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {epoch_train_loss:.4f}, '
                  f'Val MSE: {epoch_val_mse:.4f}')

    # Load the best model weights
    model.load_state_dict(best_model_state)

    # Save the model
    model_name = '_'.join(wavelets)
    torch.save(model.state_dict(), f'best_model_{model_name}.pth')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_val)
        metrics = calculate_metrics(y_val, y_pred)

    print(f"\nValidation Metrics for wavelets: {', '.join(wavelets)}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics


# Prepare the dataset for PyTorch
def create_sequences(data, context_size):
    sequences = []
    targets = []
    for i in range(len(data) - context_size):
        sequences.append(data[i:i+context_size])
        targets.append(data[i+context_size])
    return np.array(sequences), np.array(targets)

# Create sequences and targets
X, y = create_sequences(time_series_scaled, context_size)
split_index = len(X) - context_size

# Split the data: the training set includes everything up to the split index,
# and the validation set includes the last context_size observations
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
# Split the data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# # Convert the data to PyTorch tensors
# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
# X_val = torch.tensor(X_val, dtype=torch.float32)
# y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
# Create DataLoader for batch processing
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model
input_size = 1
hidden_size = 128
num_layers = 2
model = TimeSeriesModel(input_size, hidden_size, num_layers)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50
train_losses = []
val_mse_scores = []

best_val_mse = float('inf')
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        X_batch = X_batch.unsqueeze(-1)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * X_batch.size(0)  # accumulate total loss

    # Compute average training loss for this epoch
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # Evaluate on validation set
    model.eval()
    running_val_mse = 0.0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val = X_val.unsqueeze(-1)
            # print(X_val.shape)
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            running_val_mse += val_loss.item() * X_val.size(0)

    # Compute average validation MSE
    epoch_val_mse = running_val_mse / len(val_loader.dataset)
    val_mse_scores.append(epoch_val_mse)

    # Check if this is the best model so far
    if epoch_val_mse < best_val_mse:
        best_val_mse = epoch_val_mse
        best_model_state = model.state_dict()

    # Print progress
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {epoch_train_loss:.4f}, '
              f'Val MSE: {epoch_val_mse:.4f}')

# Evaluation function to calculate metrics
def calculate_metrics(y_true, y_pred):
    # Denormalize the values
    y_true = scaler.inverse_transform(y_true.detach().numpy())
    y_pred = scaler.inverse_transform(y_pred.detach().numpy())

    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))

    # MASE calculation
    naive_forecast = np.roll(y_true, 1)  # Shifted series for naive forecast
    naive_error = np.mean(np.abs(y_true[1:] - naive_forecast[1:]))
    mase = mae / naive_error

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'SMAPE': smape,
        'MASE': mase
    }

# After training, save the best model weights
model.load_state_dict(best_model_state)
torch.save(model.state_dict(), 'best_model.pth')

# Evaluate base LSTM model
print("\n===== Base LSTM Model Metrics =====")
model.eval()
with torch.no_grad():
    all_y_pred = []
    all_y_val = []
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.unsqueeze(-1)
        y_pred = model(X_batch)
        all_y_pred.append(y_pred)
        all_y_val.append(y_batch)

    y_pred = torch.cat(all_y_pred)
    y_val_combined = torch.cat(all_y_val)
    base_metrics = calculate_metrics(y_val_combined, y_pred)

print("\nBase LSTM Model Metrics:")
for metric, value in base_metrics.items():
    print(f"{metric}: {value:.4f}")
print("\n" + "=" * 30 + "\n")

# Visualize the training loss and validation MSE over epochs
epochs = range(1, num_epochs+1)
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, val_mse_scores, label='Validation MSE', color='orange')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Validation MSE over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

# Define metric names for comparison
metric_names = ['MAE', 'MSE', 'RMSE', 'SMAPE', 'MASE']

# Dictionary to store results for each combination
combination_results = {}
combination_results['base'] = base_metrics

# Train and evaluate models with different wavelet combinations
wavelet_combinations = [
    ['db1', 'sym2'],
    ['db1', 'sym2', 'dmey'],
    ['db1', 'sym2', 'haar'],
    ['db1', 'sym2', 'dmey', 'haar']
]

# Evaluate each combination
for wavelets in wavelet_combinations:
    metrics = train_evaluate_multiple_wavelets(time_series_scaled, wavelets, context_size)
    combination_results['_'.join(wavelets)] = metrics

# Compare results across combinations
print("\n===== Comparison of Wavelet Combinations =====\n")
for metric in metric_names:
    print(f"\n{metric} comparison:")
    sorted_combinations = sorted(combination_results.items(), key=lambda x: x[1][metric])
    for combination, metrics in sorted_combinations:
        print(f"{combination}: {metrics[metric]:.4f}")

# Identify best combination for each metric
print("\nBest Combination for Each Metric:")
for metric in metric_names:
    best_combination = min(combination_results.items(), key=lambda x: x[1][metric])[0]
    print(f"{metric}: {best_combination} ({combination_results[best_combination][metric]:.4f})")

# Plot predictions from each model
plt.figure(figsize=(15, 10))

# Get actual values
actual_values = scaler.inverse_transform(y_val_combined.detach().numpy())

# Plot actual values
plt.plot(range(len(actual_values)), actual_values, label='Actual Values', color='black', alpha=0.7)

# Plot predictions for each model
colors = ['blue', 'green', 'red', 'purple', 'orange']
for (name, metrics), color in zip(combination_results.items(), colors):
    # Set the correct input size for each model
    if name == 'base':
        input_size = 1
    else:
        wavelets = name.split('_')
        input_size = len(wavelets) + 1

    # Reinitialize model with correct input size
    model = TimeSeriesModel(input_size, hidden_size, num_layers)

    if name == 'base':
        # Load and use base model
        model.load_state_dict(torch.load('best_model.pth'))
    else:
        # Load and use wavelet model
        model.load_state_dict(torch.load(f'best_model_{name}.pth'))

    model.eval()
    with torch.no_grad():
        all_y_pred = []
        for X_batch, _ in val_loader:
            if name == 'base':
                X_batch = X_batch.unsqueeze(-1)
            else:
                # Apply wavelet transforms for non-base models
                wavelets = name.split('_')
                X_wavelet = apply_multiple_wavelets_to_sequences(X_batch.numpy(), wavelets)
                X_batch = torch.tensor(X_wavelet, dtype=torch.float32)

            y_pred = model(X_batch)
            all_y_pred.append(y_pred)

        predictions = torch.cat(all_y_pred)
        predictions = scaler.inverse_transform(predictions.detach().numpy())
        plt.plot(range(len(predictions)), predictions, label=f'{name} Model', color=color, alpha=0.6)

plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.title('Model Predictions Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
