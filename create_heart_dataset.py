# create_heart_dataset.py
# This creates a sample heart.csv file if you don't have one

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate synthetic heart disease dataset
data = {
    'Age': np.random.randint(29, 80, n_samples),
    'Sex': np.random.choice(['M', 'F'], n_samples, p=[0.6, 0.4]),
    'ChestPainType': np.random.choice(['ATA', 'NAP', 'ASY', 'TA'], n_samples),
    'RestingBP': np.random.normal(130, 20, n_samples).astype(int),
    'Cholesterol': np.random.normal(240, 50, n_samples).astype(int),
    'FastingBS': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    'RestingECG': np.random.choice(['Normal', 'ST', 'LVH'], n_samples, p=[0.6, 0.3, 0.1]),
    'MaxHR': np.random.normal(150, 25, n_samples).astype(int),
    'ExerciseAngina': np.random.choice(['N', 'Y'], n_samples, p=[0.7, 0.3]),
    'Oldpeak': np.random.exponential(1, n_samples).round(1),
    'ST_Slope': np.random.choice(['Up', 'Flat', 'Down'], n_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Clip values to realistic ranges
df['RestingBP'] = np.clip(df['RestingBP'], 90, 200)
df['Cholesterol'] = np.clip(df['Cholesterol'], 120, 600)
df['MaxHR'] = np.clip(df['MaxHR'], 70, 202)
df['Oldpeak'] = np.clip(df['Oldpeak'], 0, 6.2)

# Create target variable with realistic correlations
target_prob = (
    0.1 * (df['Age'] > 55) +
    0.1 * (df['Sex'] == 'M') +
    0.15 * (df['ChestPainType'] == 'ASY') +
    0.1 * (df['RestingBP'] > 140) +
    0.1 * (df['Cholesterol'] > 240) +
    0.1 * (df['FastingBS'] == 1) +
    0.1 * (df['RestingECG'] != 'Normal') +
    0.1 * (df['MaxHR'] < 120) +
    0.2 * (df['ExerciseAngina'] == 'Y') +
    0.1 * (df['Oldpeak'] > 2) +
    0.1 * (df['ST_Slope'] == 'Down') +
    np.random.normal(0, 0.1, n_samples)
)

df['HeartDisease'] = (target_prob > 0.5).astype(int)

# Save to CSV
df.to_csv('heart.csv', index=False)
print("✅ heart.csv dataset created successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Heart disease cases: {df['HeartDisease'].sum()}/{len(df)} ({df['HeartDisease'].mean():.1%})")
print("\nFirst 5 rows:")
print(df.head())