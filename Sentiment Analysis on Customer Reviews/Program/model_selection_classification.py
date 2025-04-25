import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def model_selection_classification(X, y):
    # Splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models that need feature scaling
    scale_needed = {'Logistic Regression', 'Support Vector Classifier', 'K-Nearest Neighbors'}

    # Standardize features if needed
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Popular Classification Models
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Support Vector Classifier': SVC(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'XGBoost': XGBClassifier()
    }

    results = []

    for name, model in models.items():
        try:
            # Use scaled data for models that need it
            if name in scale_needed:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # Compute Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            results.append([name, accuracy, precision, recall, f1])

        except Exception as e:
            print(f"⚠️ {name} failed: {e}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    results_df = results_df.sort_values(by='Accuracy', ascending=False)  # Sort by best accuracy

    return results_df

# Example Usage:
# Assuming you have a dataset X, y
# results = model_selection_classification(X, y)
# print(results)
