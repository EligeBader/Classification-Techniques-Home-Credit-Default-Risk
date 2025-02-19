# 🏡 Home Credit Default Risk Classification Project

**Python | Machine Learning | Kaggle | Data Science**

---

### 📄 Description
Many people struggle to get loans due to insufficient or non-existent credit histories. Unfortunately, this population is often taken advantage of by untrustworthy lenders.

**Home Credit Group** strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. They use a variety of alternative data—including telco and transactional information—to predict their clients' repayment abilities. Although Home Credit currently uses various statistical and machine learning methods for these predictions, they are challenging Kagglers to help unlock the full potential of their data to ensure clients capable of repayment are not rejected and that loans are given with an appropriate principal, maturity, and repayment schedule to empower their clients' success.

### 📊 Evaluation
Submissions are evaluated on the area under the ROC curve (AUC) between the predicted probability and the observed target.

### 📁 Dataset Description
- **application_{train|test}.csv:** Main table, static data for all applications. One row represents one loan.
- **bureau.csv:** Previous credits from other financial institutions reported to Credit Bureau.
- **bureau_balance.csv:** Monthly balances of previous credits in Credit Bureau.
- **POS_CASH_balance.csv:** Monthly balance snapshots of previous POS (point of sales) and cash loans from Home Credit.
- **credit_card_balance.csv:** Monthly balance snapshots of previous credit cards from Home Credit.
- **previous_application.csv:** All previous applications for Home Credit loans.
- **installments_payments.csv:** Repayment history for previously disbursed credits.
- **HomeCredit_columns_description.csv:** Column descriptions for the datasets.

### 🛠 Tools & Technologies
- **Python 3.8+:** Primary programming language
- **Pandas:** Data manipulation and analysis
- **Scikit-learn:** Machine learning algorithms and tools
- **TensorFlow & Keras:** Neural network models
- **Imbalanced-learn:** Handling imbalanced datasets
- **Jupyter Notebook:** Interactive coding and exploration

### 🔄 Workflow

1. **Libraries and Data Loading:**
    - Imported the necessary libraries for data manipulation, machine learning, and neural networks.
    - Loaded the dataset using functions to read CSV files.

2. **Data Preprocessing:**
    - **Dropping Unnecessary Features:** Removed columns that were not needed for the analysis to streamline the dataset.
    - **Handling Missing Values:** Utilized imputation techniques to fill in missing values in both numerical and categorical data.
    - **Encoding Categorical Data:** Transformed categorical features into numerical representations using encoders.

3. **Data Splitting:**
    - Split the data into training and testing sets to build and evaluate models. Handled imbalanced data using undersampling techniques to balance the classes.

4. **Model Training:**
    - **Logistic Regression Model:** Trained a logistic regression model, a simple and interpretable model for binary classification.
    - **Random Forest Model:** Trained a random forest model, an ensemble learning method that combines multiple decision trees.
    - **XGBoost Model:** Trained an XGBoost model, a robust and efficient gradient boosting algorithm.
    - **Neural Network Model:** Built and trained a neural network model using Keras, experimenting with different architectures and parameters.

5. **Prediction:**
    - Made predictions on the test set using the trained models. Generated probabilities for the target variable (default risk).

6. **Evaluation:**
    - Evaluated the models using metrics like ROC-AUC, accuracy, precision, recall, and F1-score to determine their performance.

### 📂 Project Structure

```
- home_credit_default_risk
  - data/
    - application_train.csv
    - application_test.csv
  - models/
    - trained_logistic_model.pickle
    - trained_random_forest_model.pickle
    - trained_xgb_model.pickle
    - trained_nn_model.pickle
  - notebooks/
    - training_class.ipynb
    - testing_class.ipynb
  - scripts/
    - helper_functions.py
  - submission/
    - predictions_lr.csv
    - predictions_rf.csv
    - predictions_xgb.csv
    - predictions_nn.csv
  - README.md
```

### 🧩 Models and Results

- **Logistic Regression:**
    - Simple and interpretable model.
    - Moderate performance with good baseline metrics.

- **Random Forest:**
    - Ensemble method combining multiple decision trees.
    - Improved performance with higher accuracy and ROC-AUC compared to logistic regression.

- **XGBoost:**
    - Advanced gradient boosting algorithm.
    - Provided strong predictive power and robust performance.

- **Neural Network:**
    - Flexible and powerful model using deep learning techniques.
    - Achieved high accuracy and ROC-AUC with optimized architecture and parameters.

### 🎯 Best Model
Based on the evaluation metrics, the **XGBoost** model provided the best performance in terms of ROC-AUC and overall predictive power.

### 🌟 Improvements
To enhance this project further, consider:
- Experimenting with different models (e.g., Gradient Boosting).
- Optimizing hyperparameters with grid search or random search.
- Incorporating more advanced feature engineering techniques.
