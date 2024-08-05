import pandas as pd
import optuna
import mlflow
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('data/winequality-red.csv')

# Features and target variable
X = df.drop('quality', axis=1)
y = df['quality']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Optuna objective function
def objective(trial):
    print("Another trial:",trial)
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    
    return accuracy

# Create Optuna study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Print best parameters and accuracy
print("Best parameters: ", study.best_params)
print("Best accuracy: ", study.best_value)

# Log experiment with MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.start_run()
mlflow.log_params(study.best_params)
mlflow.log_metric("accuracy", study.best_value)

# Train final model with best parameters and save it
best_params = study.best_params
final_model = RandomForestClassifier(**best_params)
final_model.fit(X_train, y_train)
joblib.dump(final_model, 'model.pkl')

mlflow.sklearn.log_model(final_model, "model")
mlflow.end_run()