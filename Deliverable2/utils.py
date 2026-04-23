from statsmodels.tsa.stattools import adfuller

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
