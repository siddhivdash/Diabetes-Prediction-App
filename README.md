# 🧠 Diabetes Prediction using Machine Learning

This project is a Flask-based web application that predicts whether a person is diabetic or non-diabetic using medical attributes and a trained machine learning model.

---

## 📌 Objective

To build a predictive model using supervised machine learning algorithms that can classify individuals as **diabetic** or **non-diabetic** based on their health parameters.

---

## 📁 Dataset Used

- **Source**: Pima Indians Diabetes Dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Features**:
  - `Pregnancies`
  - `Glucose`
  - `BloodPressure`
  - `SkinThickness`
  - `Insulin`
  - `BMI`
  - `DiabetesPedigreeFunction`
  - `Age`
- **Target**: `Outcome` (0 = Non-Diabetic, 1 = Diabetic)

---

## 🧪 Models Explored

Multiple regression-based classification models were tested and evaluated:

### 1. **Linear Regression (Not Used)**
- Initially tried for comparison.
- **Reason for exclusion**: Not ideal for classification tasks; outputs continuous values instead of class labels.

---

### 2. **Lasso Regression**
- A linear model with L1 regularization.
- Helps with feature selection by shrinking coefficients.
- **Result**: Slight underfitting and lower accuracy compared to Ridge and Logistic Regression.

---

### 3. **Ridge Regression**
- Adds L2 penalty to reduce model complexity.
- Used as an intermediary benchmark model.
- **Result**: Stable performance but still not ideal for binary classification.

---

### 4. **Elastic Net**
- Combination of L1 and L2 regularization.
- Useful in high-dimensional settings.
- **Result**: Good but marginally less accurate than Logistic Regression.

---

### ✅ 5. **Logistic Regression (Final Model)**
- A classification algorithm best suited for binary outcomes.
- Provides probabilistic predictions for class 0 or 1.
- **Selected as the final model** based on:
  - Highest accuracy and F1-score
  - Simplicity and interpretability
  - Fast training and prediction time

---

## 🔍 Hyperparameter Tuning using GridSearchCV

- During model evaluation, **`GridSearchCV`** was used to perform exhaustive search over specified parameter values for each estimator.
- Cross-validation was applied to tune:
  - **Lasso Regression**: `alpha`
  - **Ridge Regression**: `alpha`
  - **Elastic Net**: `l1_ratio` and `alpha`
  - **Logistic Regression**: `C` (inverse of regularization strength), `penalty`, and `solver`

**Example GridSearch for Logistic Regression**:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1.0, 10.0],
    'penalty': ['l2'],
    'solver': ['liblinear', 'saga']
}

grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_scaled, y_train)

print("Best Parameters:", grid.best_params_)
