from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np

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