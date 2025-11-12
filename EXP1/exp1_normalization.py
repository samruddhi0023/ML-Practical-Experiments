
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

data = {
    'Age': [18, 22, 30, 45, 52, 60],
    'Salary': [10000, 20000, 35000, 50000, 65000, 80000],
    'Experience': [1, 3, 5, 8, 10, 15]
}

df = pd.DataFrame(data)
print("Original Dataset:\n", df)

minmax_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(minmax_scaler.fit_transform(df), columns=df.columns)
print("\nMin-Max Normalized Data (0 to 1 range):\n", df_minmax)

standard_scaler = StandardScaler()
df_standard = pd.DataFrame(standard_scaler.fit_transform(df), columns=df.columns)
print("\nStandardized Data (Mean = 0, Std = 1):\n", df_standard)

def decimal_scaling(df):
    df_scaled = df.copy()
    for col in df.columns:
        max_val = df[col].abs().max()
        j = np.ceil(np.log10(max_val + 1))
        df_scaled[col] = df[col] / (10 ** j)
    return df_scaled

df_decimal = decimal_scaling(df)
print("\nDecimal Scaled Data:\n", df_decimal)

df_log = np.log1p(df)
print("\nLogarithmic Transformed Data:\n", df_log)

normalizer = Normalizer()
df_unitvector = pd.DataFrame(normalizer.fit_transform(df), columns=df.columns)
print("\nUnit Vector Normalized Data:\n", df_unitvector)

print("\nConclusion: Data normalization has been successfully performed using multiple techniques.")
