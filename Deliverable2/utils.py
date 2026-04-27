from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

def perform_adf_test(series, name):
    """
    Realiza el test de Dickey-Fuller Aumentado y muestra los resultados interpretados.
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






def plot_model_comparison_styled(actual, models_setup, last_n=100):
    """
    actual: Serie de Pandas con los retornos reales.
    models_setup: Diccionario de diccionarios con la configuración:
                  { 'Nombre': {'data': serie, 'color': 'red', 'ls': '--', 'lw': 2} }
    """
    plt.figure(figsize=(15, 6))
    
    # 1. Realidad del mercado (siempre como base)
    plt.plot(actual.tail(last_n).values, label='Actual Returns', color='black', alpha=0.2, lw=1)
    
    # 2. Iterar sobre la configuración de cada modelo
    for name, config in models_setup.items():
        # Calcular Hit Ratio para el label
        hit_ratio = (np.sign(actual) == np.sign(config['data'])).mean() * 100
        
        plt.plot(
            config['data'].tail(last_n).values, 
            label=f"{name} (Hit: {hit_ratio:.1f}%)",
            color=config.get('color', 'blue'),
            linestyle=config.get('ls', '-'),
            linewidth=config.get('lw', 2),
            alpha=config.get('alpha', 0.8)
        )

    plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
    plt.title(f'Advanced Model Comparison (Last {last_n} days)', fontsize=14)
    plt.xlabel('Time Steps')
    plt.ylabel('Returns (%)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()
    
    
def comparison_plot(actual, models_setup, last_n=100, use_zoom=True, color = 'blue'):
    fig, ax1 = plt.subplots(figsize=(15, 7))
    
    # 1. Realidad del mercado (Eje izquierdo siempre)
    actual_data = actual.tail(last_n).values
    ax1.plot(actual_data, label='Actual Returns', color='black', alpha=0.15, lw=1)
    ax1.set_ylabel('Market Returns (%)', color='black')
    
    # Decidimos si creamos un segundo eje o usamos el primero
    ax_models = ax1.twinx() if use_zoom else ax1
    
    if use_zoom:
        ax_models.set_ylabel('Models Prediction Scale (%)', color=color, fontsize=12)
    
    # 2. Iterar sobre los modelos
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

    # Configuración de leyendas (unificando ambos ejes si hace falta)
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
    
    
    
    
    

def train_lstm_model(model, train_loader, val_loader, criterion, optimizer, 
                     device="cpu", epochs=100, patience=10, save_path="best_lstm.pth",
                     log_dir="runs/apple_lstm_experiment"):
    """
    Entrena la LSTM para regresión con Early Stopping y soporte para TensorBoard.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Inicializamos el escritor de TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    
    model.to(device)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        # ---- Entrenamiento ----
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # squeeze() para asegurar que las dimensiones coincidan (batch, 1) vs (batch)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_y.size(0)

        train_loss /= len(train_loader.dataset)

        # ---- Validación ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item() * batch_y.size(0)

        val_loss /= len(val_loader.dataset)

        # ---- Guardar en TensorBoard ----
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        # También podemos loguear el learning rate si usamos schedulers
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

    # Cerrar TensorBoard y cargar mejor modelo
    writer.close()
    model.load_state_dict(torch.load(save_path))
    return model




def get_lstm_predictions(model, X_tensor, scaler, original_index, window_size):
    """
    Transforma los tensores en predicciones reales y alineadas con sus fechas.
    """
    model.eval()
    with torch.no_grad():
        # 1. Predicción (la pasamos a CPU y Numpy)
        preds = model(X_tensor).cpu().numpy()
    
    # 2. El "Truco" del Scaler: Crear matriz temporal de 5 columnas
    # (Usamos el mismo número de columnas con el que entrenamos el scaler)
    placeholder = np.zeros((len(preds), scaler.n_features_in_))
    placeholder[:, 0] = preds.flatten()
    
    # 3. Inverse Transform
    unscaled = scaler.inverse_transform(placeholder)[:, 0]
    
    # 4. Alinear con el índice original (quitando el desfase de la ventana)
    # Importante: El índice empieza en window_size porque los primeros días se "pierden" para crear la secuencia
    final_series = pd.Series(unscaled, index=original_index[window_size:])
    
    return final_series