# Hyperparameter Optimization (HPO)

## Objective:
In this lab, you will learn to use [Optuna](https://optuna.org/) to fine-tune a [CatBoost](https://catboost.ai/docs/en/concepts/python-quickstart) model on the [Covertype Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html). The objective is to maximize classification accuracy by finding the best hyperparameter combination for this multiclass classification problem.

## 1. Installing Necessary Packages

To get started, install the required Python packages by running the following command in your terminal or command prompt:

```bash
pip install optuna catboost pandas scikit-learn optuna-dashboard seaborn matplotlib
```

What’s Installed:
- Optuna: For hyperparameter optimization.
- CatBoost: The gradient boosting library we’ll tune.
- Pandas: For data manipulation.
- Scikit-learn: For dataset loading and evaluation metrics.
- Optuna-dashboard: For real-time monitoring of optimization.
- Seaborn & Matplotlib: For data visualizations.

## 2. Loading the Covertype Dataset
The Covertype dataset is a multiclass classification task with 54 features, 7 classes, and over 500,000 samples. We’ll load it directly using Scikit-learn’s `fetch_covtype` function.

```python
from sklearn.datasets import fetch_covtype
import pandas as pd

# Load the dataset
covtype = fetch_covtype(as_frame=True)  # Returns a DataFrame
X = covtype.data  # Features
y = covtype.target - 1  # Target (adjusted to 0-6 for zero-based indexing)
```

Why Covertype?
- Multiclass Complexity: Predicts 7 forest cover types.
- Mixed Features: Includes both numerical and categorical (binary) features, perfect for CatBoost.
- Large Scale: Tests model performance on a substantial dataset.

To make computations faster, we'll take into account only 20k samples.
```python
sample_size = 20000
# Ensure we get a balanced sample across all classes
from sklearn.model_selection import train_test_split
X_sample, _, y_sample, _ = train_test_split(
    X, y, train_size=sample_size, random_state=42, stratify=y
)
X = X_sample
y = y_sample
```

## 3. Visualizing the Dataset
Visualizations help us understand the data before modeling. Below are three useful plots:

### Class Distribution
Examine the balance of classes in the target variable.
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x=y)
plt.title('Class Distribution in Covertype Dataset')
plt.xlabel('Cover Type')
plt.ylabel('Count')
plt.show()
```

### Feature Correlation Heatmap
Check correlations among the first 10 features.
```python
# Select first 10 features for simplicity
subset = X.iloc[:, :10]

# Compute correlation matrix
corr = subset.corr()

# Plot heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of First 10 Features')
plt.show()
```

### Pairplot for Selected Features
Visualize relationships between the first 3 features and class separation.

```python
# Combine features and target for a sample
sample = X.iloc[:, :3].copy()  # First 3 features
sample['Cover Type'] = y

sns.pairplot(sample, hue='Cover Type', diag_kind='kde')
plt.show()
```

## 4. Setting up Optuna Dashboard
Optuna offers a dashboard to monitor optimization in real-time. Launch it with:

```bash
optuna-dashboard sqlite:///example.db
```

## 5. Train and evaluate CatBoost with default settings

Before performing hyperparameter optimization, it’s important to establish a baseline model using CatBoost with default settings. This helps us gauge how much improvement is achieved through tuning.

```python
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the dataset into training, validation and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Initialize the CatBoost model with default settings
model = CatBoostClassifier(verbose=0, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
baseline_accuracy = accuracy_score(y_test, y_pred)

print(f"Baseline Accuracy: {baseline_accuracy * 100:.2f}%")
```

Confusion matrices are great for understanding the model's performance on a per-class basis. They provide detailed insight into which classes are being confused with each other, allowing you to identify weaknesses in the model’s predictions.

```python
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title('Confusion Matrix for Baseline Model')
plt.show()
``` 

## 6. Implementing Hyperparameter Tuning with Optuna
The objective function drives the optimization:
- Takes a trial object from Optuna.
- Suggests hyperparameter values.
- Trains a CatBoost model.
- Evaluates accuracy on the validation set.
- Returns the score (accuracy).

This [notebook](https://github.com/optuna/optuna-examples/blob/main/quickstart.ipynb) provides a useful example of implementing Optuna, although it employs cross-validation (CV) rather than a train/validation split. Additionally, the CatBoost documentation [page](https://catboost.ai/docs/en/concepts/parameter-tuning) offers comprehensive guidance on parameter tuning.
```python
import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        # TODO: Investigate CatBoost docs and add more hyperparameters
    }

    # TODO: 
    # 1. Train CatBoost using params
    # 2. Return validation accuracy score as result
    # Area for improvement: use Cross-Validation instead of a single train/val split.
```

Set up and run the Optuna study:
```python
# Create the study
study_name = 'catboost_optimization'
storage_name = 'sqlite:///example.db' 
study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    direction='maximize',
    load_if_exists=True
)
study.optimize(objective, n_trials=20)  # Run 20 trials
```
Monitor trials at http://localhost:8080. Discuss results. Which features are most important?

## 7. Post-Optimization Steps: Analyzing Results and Final Evaluation

After the Optuna study completes, we need to analyze the results, train a final model using the best hyperparameters found, and evaluate its performance on the unseen test set.

```python
# Retrieve the best trial from the study
best_trial = study.best_trial

print(f"Best Trial Number: {best_trial.number}")
print(f"Best Validation Accuracy: {best_trial.value:.4f}")
print("Best Hyperparameters Found:")
for key, value in best_trial.params.items():
    print(f"  {key}: {value}")
```

> Note: Review the best hyperparameters found. Do they make sense? Are any parameters hitting the boundaries of their search space (e.g., iterations consistently at 1000)? If so, you might consider expanding the search space in a future run. The "Best Validation Accuracy" is the score Optuna maximized using the validation set within the objective function.

Now, we train a new CatBoost model using the best hyperparameters found by Optuna. Crucially, we train this model on the combination of the original training data and validation data. This allows the final model to learn from more data than any single model trained during the optimization trials.

```python
best_params = best_trial.params

# --- IMPORTANT: Combine Training and Validation Data ---
# Create the full training dataset (train + validation) for the final model
X_train_full = pd.concat([X_train, X_valid], ignore_index=True)
y_train_full = pd.concat([y_train, y_valid], ignore_index=True)

final_params = best_params.copy()
final_params['random_state'] = 42

final_model = CatBoostClassifier(**final_params)
final_model.fit(X_train_full, y_train_full)
```

This is the moment of truth! We evaluate our optimized `final_model` on the completely unseen X_test and y_test to get an unbiased estimate of its generalization performance. We then compare this to the baseline model's performance on the same test set.

```python
y_pred_test = final_model.predict(X_test)

# Calculate final accuracy
final_accuracy = accuracy_score(y_test, y_pred_test)

print("--- Performance Comparison ---")
# Assuming baseline_accuracy variable holds the score from Step 5
print(f"Baseline Accuracy (on Test Set): {baseline_accuracy * 100:.2f}%")
print(f"Optimized Accuracy (on Test Set): {final_accuracy * 100:.2f}%")
improvement = final_accuracy - baseline_accuracy
print(f"Improvement due to HPO: {improvement * 100:.2f}%")

print("Confusion Matrix for Final Optimized Model (on Test Set):")
ConfusionMatrixDisplay.from_estimator(final_model, X_test, y_test)
plt.title('Confusion Matrix for Final Optimized Model')
plt.tight_layout()
plt.show()
```

Finally, let's examine which features the optimized CatBoost model found most important for making predictions.

```python
importances = final_model.get_feature_importance()
feature_names = X_train_full.columns

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances (e.g., top 20)
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20)) # Plot top 20
plt.title('Top 20 Feature Importances from Final Optimized Model')
plt.tight_layout()
plt.show()
```