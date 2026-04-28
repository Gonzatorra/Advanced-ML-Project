from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

def perform_adf_test(series, name):
    """
    Performs the Augmented Dickey-Fuller test and prints the interpreted results.

    This function tests if a time series is stationary. It automatically handles 
    missing values by dropping NaNs before the calculation.

    Args:
        series (pd.Series): The time series data to be tested.
        name (str): A descriptive name for the series, used for display purposes.

    Returns:
        None: The function prints the results directly to the console.
    """
    # Make sure to drop NaN values for the test
    result = adfuller(series.dropna())
    
    print(f'--- RESULTS OF DICKEY-FULLER TEST FOR: {name.upper()} ---')
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4e}')
    
    # Automatic interpretation
    if result[1] < 0.05:
        print(f"\nREJECT the null hypothesis: The series '{name}' is STATIONARY.")
        print("The data is ready for the ARIMA model.")
    else:
        print(f"\nFAIL to reject the null hypothesis: The series '{name}' is NON-STATIONARY.")
        print("Transformation (like differencing) is required.")
    print('-' * 60 + '\n')





def calculate_hit_rate(y_true, y_pred):
    """
    Calculates the percentage of correct direction predictions (Hit Rate).

    This metric measures how often the predicted sign (positive or negative) 
    matches the actual sign of the returns.

    Args:
        y_true (array-like): The actual observed values.
        y_pred (array-like): The values predicted by the model.

    Returns:
        float: The hit rate as a percentage (0 to 100).
    """
    hits = (np.sign(y_true) == np.sign(y_pred)).sum()
    return (hits / len(y_true)) * 100










# ------------- #
# --- PLOTS --- #
# ------------- #  
def comparison_plot(actual, models_setup, last_n=100, use_zoom=True, color = 'blue'):
    """
    Plots market returns against model predictions with an optional secondary Y-axis.

    Args:
        actual (pd.Series): The real market returns.
        models_setup (dict): Configuration for models, including 'data', 'color', 
            'ls' (linestyle), and 'lw' (linewidth).
        last_n (int, optional): Number of recent data points to show. Defaults to 100.
        use_zoom (bool, optional): If True, plots models on a secondary Y-axis to 
            better visualize small movements. Defaults to True.
        color (str, optional): Label color for the secondary axis. Defaults to 'blue'.

    Returns:
        None: Displays a Matplotlib figure.
    """
    fig, ax1 = plt.subplots(figsize=(15, 7))
    
    # Market reality (always on the left axis)
    actual_data = actual.tail(last_n).values
    ax1.plot(actual_data, label='Actual Returns', color='black', alpha=0.15, lw=1)
    ax1.set_ylabel('Market Returns (%)', color='black')
    
    # Decide if we use a secondary axis for models
    ax_models = ax1.twinx() if use_zoom else ax1
    
    if use_zoom:
        ax_models.set_ylabel('Models Prediction Scale (%)', color=color, fontsize=12)
    
    # Iterate over models
    for name, config in models_setup.items():
        # Hit Ratio para el label
        hit_ratio = (np.sign(actual) == np.sign(config['data'])).mean() * 100
        
        ax_models.plot(
            config['data'].tail(last_n).values, 
            label=f"{name} (Hit: {hit_ratio:.1f}%)",
            color=config.get('color', 'blue'),
            linestyle=config.get('ls', '-'),
            linewidth=config.get('lw', 2),
            alpha=config.get('alpha', 0.8)
        )

    # Legend handling
    lines1, labels1 = ax1.get_legend_handles_labels()
    if use_zoom:
        lines2, labels2 = ax_models.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True, shadow=True)
    else:
        ax1.legend(loc='upper right')

    title_type = "(Double Axis Zoom)" if use_zoom else "(Standard Scale)"
    plt.title(f'Market vs Models {title_type} - Last {last_n} Days', fontsize=14)
    plt.grid(True, alpha=0.2)
    fig.tight_layout()
    plt.show()
    
    
    
    
    





# ------------ #
# --- LSTM --- #
# ------------ #  
def train_lstm_model(model, train_loader, val_loader, criterion, optimizer, 
                     device="cpu", epochs=100, patience=10, save_path="best_lstm.pth",
                     log_dir="runs/apple_lstm_experiment"):
    """
    Trains an LSTM model for regression with Early Stopping and TensorBoard logging.

    Args:
        model (nn.Module): The PyTorch LSTM model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion (callable): Loss function (e.g., nn.MSELoss).
        optimizer (Optimizer): PyTorch optimizer (e.g., Adam).
        device (str, optional): Device to run on ('cpu' or 'cuda'). Defaults to "cpu".
        epochs (int, optional): Maximum number of training iterations. Defaults to 100.
        patience (int, optional): Epochs to wait for improvement before Early Stopping. Defaults to 10.
        save_path (str, optional): Path to save the best model weights. Defaults to "best_lstm.pth".
        log_dir (str, optional): Directory for TensorBoard logs.

    Returns:
        nn.Module: The model with the best weights loaded.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    model.to(device)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_y.size(0)

        train_loss /= len(train_loader.dataset)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item() * batch_y.size(0)

        val_loss /= len(val_loader.dataset)

        # ---- Save to TensorBoard ----
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        # We can also log the learning rate if we use schedulers
        writer.add_scalar('Params/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # ---- Early Stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping en época {epoch+1}")
            break

    # Close TensorBoard and load the best model weights
    writer.close()
    model.load_state_dict(torch.load(save_path))
    return model





def get_lstm_predictions(model, X_tensor, scaler, original_index, window_size):
    """
    Generates model predictions and transforms them back to the original scale.

    Args:
        model (nn.Module): The trained LSTM model.
        X_tensor (torch.Tensor): The input features as a tensor.
        scaler (sklearn.preprocessing): The fitted scaler used during training.
        original_index (pd.Index): The original dates/index of the dataframe.
        window_size (int): The number of time steps used for each prediction.

    Returns:
        pd.Series: The unscaled predictions aligned with the correct dates.
    """
    model.eval()
    with torch.no_grad():
        # Compute the predictions in the scaled space
        preds = model(X_tensor).cpu().numpy()
    
    # Use the same approach as before to inverse transform the predictions
    # Create a placeholder array with the same number of features as the scaler expects
    placeholder = np.zeros((len(preds), scaler.n_features_in_))
    placeholder[:, 0] = preds.flatten()
    
    # Inverse transform to get predictions in the original scale
    unscaled = scaler.inverse_transform(placeholder)[:, 0]
    
    # 4. Align with the original index (removing the window offset)
    # Important: The index starts at window_size because the first days are lost when creating the sequence
    final_series = pd.Series(unscaled, index=original_index[window_size:])
    
    return final_series