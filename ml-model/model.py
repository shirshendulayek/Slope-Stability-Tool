#for webapp
import pandas as pd
import numpy as np
!pip install optuna
import optuna
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv('/content/merged_data15.csv')

print(df.info())
# Drop Reference and empty columns
#df = df.drop(columns=['Reference', 'Unnamed: 6', 'Unnamed: 7'], errors='ignore')
df = df.dropna()

# Define features and target
X = df.drop(columns=['FoS']).astype(float)
y = df['FoS'].astype(float)



# Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Bayesian Optimization Objectives ---

def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
    }
    model = RandomForestRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    return r2_score(y_test, model.predict(X_test))

def objective_gbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10)
    }
    model = GradientBoostingRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    return r2_score(y_test, model.predict(X_test))

def objective_dl(trial):
    # DL model architecture search
    n_layers = trial.suggest_int('n_layers', 1, 3)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(trial.suggest_int('units_0', 16, 128), activation='relu', input_shape=(X.shape[1],)))
    for i in range(1, n_layers):
        model.add(tf.keras.layers.Dense(trial.suggest_int(f'units_{i}', 16, 128), activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_scaled, y_train, epochs=50, verbose=0)

    preds = model.predict(X_test_scaled)
    return r2_score(y_test, preds)

# --- Execute Optimization ---

print("Optimizing Random Forest...")
study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=20)

print("Optimizing Gradient Boosting...")
study_gbm = optuna.create_study(direction='maximize')
study_gbm.optimize(objective_gbm, n_trials=20)

print("Optimizing Deep Learning...")
study_dl = optuna.create_study(direction='maximize')
study_dl.optimize(objective_dl, n_trials=10)

print("\n--- Best R2 Scores ---")
print(f"Random Forest: {study_rf.best_value:.4f}")
print(f"Gradient Boosting: {study_gbm.best_value:.4f}")
print(f"Deep Learning: {study_dl.best_value:.4f}")

pickle.dump(study_rf, open('model.pkl', 'wb'))

print(study_rf.predict([19, 30, 26, 30, 38]))
