''' 
In this notebook, we performed data preprocessing, including loading data, handling missing values, feature encoding,
and data splitting, followed by training and evaluating machine learning models. The purpose of this notebook is to develop 
a robust pipeline for predictive modeling, ensuring data quality and optimizing model performance. This project aims to build,
train, and test models for accurate predictions using various preprocessing techniques and machine learning algorithms.

'''

# %%
import pandas as pd
import numpy as np
import pickle
import dill
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import TargetEncoder, OrdinalEncoder, OneHotEncoder
import imblearn
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping
import keras 
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# %%


def load_data(file):
    '''
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    file (str): The path to the CSV file.

    Returns:
    DataFrame: The loaded data as a pandas DataFrame.
     '''
    df = pd.read_parquet(file, engine='pyarrow')

    return df

with open('read_file.pickle', 'wb') as f:
    dill.dump(load_data, f)





def drop_features(df, features_to_drop=[]):

    '''
    Drop specified features from the DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame.
    features_to_drop (list): List of column names to drop.

    Returns:
    DataFrame: The DataFrame with specified features dropped.
    '''
    df = df.drop(columns=features_to_drop)

    return df

with open('drop_features.pickle', 'wb') as f:
    dill.dump(drop_features, f)




# %%
def split_data(df, target, feature_selected= None, features_dropped =[], balanced_data=True):

    '''
    Split the data into training and testing sets, with optional balancing.

    Parameters:
    df (DataFrame): The input DataFrame.
    target (str): The target column name.
    feature_selected (list): List of selected features.
    features_dropped (list): List of features to drop.
    balanced_data (bool): Whether to balance the data using undersampling.

    Returns:
    tuple: The training and testing sets.
    '''

    if balanced_data == True:
        if feature_selected == None:
            X = df.drop(columns= [target] + features_dropped)
            y = df[target]

        else:
            X = df[feature_selected]
            y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test

    else:
        if feature_selected == None:
            X = df.drop(columns= [target] + features_dropped)
            y = df[target]

        else:
            X = df[feature_selected]
            y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rus = imblearn.under_sampling.RandomUnderSampler()
        xtrain_rus, ytrain_rus = rus.fit_resample(X_train, y_train)

        return xtrain_rus, X_test, ytrain_rus, y_test


with open('split_data.pickle', 'wb') as f:
    dill.dump(split_data, f)




# %%
def clean_data(df):

    '''
    Clean the data by imputing missing values.

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    DataFrame: The cleaned DataFrame with imputed values.
    '''
    #Use SimpleImputer

    #Check Columns having nulls
    nulls_col = df.columns[df.isnull().sum() > 0]
    nulls_col  = list(nulls_col)


    # Separate numeric and categorical features
    numeric_features = [feat for feat in nulls_col if df[feat].dtype.kind in 'bifc']
    categorical_features = [feat for feat in nulls_col if feat not in numeric_features]

    # Impute missing values for numeric features
    numeric_imputer = SimpleImputer(strategy='median')
    df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])

    # Impute missing values for categorical features    
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])


    return df

with open('clean_data.pickle', 'wb') as f:
    dill.dump(clean_data, f)




# %%
def encode_data(df, target, categorical_cols, train, model):

    '''
    Encode categorical data using the specified encoding model.

    Parameters:
    df (DataFrame): The input DataFrame.
    target (str): The target column name.
    categorical_cols (list): List of categorical columns to encode.
    train (bool): Whether to train the encoder.
    model (class): The encoding model class.

    Returns:
    DataFrame: The DataFrame with encoded categorical data.
    '''

    file_name = 'trained_data.pickle'
    if not train: 
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                mod = dill.load(f)

             # Transform the categorical columns 
            tempvar = mod.transform(df[categorical_cols])
            print(type(df))
            print(type(tempvar))

            if model == TargetEncoder or model == OrdinalEncoder:
                # Update the original DataFrame with encoded values 
                for i, col in enumerate(categorical_cols): 
                    df[col] = tempvar[:, i]
            else:
                df.drop(columns = categorical_cols, axis =1 , inplace=True)
                df = pd.concat([df, tempvar], axis = 1)


    else:

        # Initialize and fit the TargetEncoder 
        mod = model() 
        if model == TargetEncoder:
            mod.fit(df[categorical_cols], df[target])
        else:
            mod.fit(df[categorical_cols])

        
        # Transform the categorical columns 
        tempvar = mod.transform(df[categorical_cols])
        print(type(df))
        print(type(tempvar))
        
        if model == TargetEncoder or model == OrdinalEncoder:
            # Update the original DataFrame with encoded values 
            for i, col in enumerate(categorical_cols): 
                df[col] = tempvar[:, i]
        else:
            df.drop(columns = categorical_cols, axis =1 , inplace=True)
            df = pd.concat([df, tempvar], axis = 1)
            
               
                
        
        with open(file_name, 'wb') as f:
            dill.dump(mod, f)

    return df


with open('encode_data.pickle', 'wb') as f:
    dill.dump(encode_data, f)



# %%
def train_model(model_class, X_train, y_train, param_grid={}, best_combination=False, n_trials=10, **args):
    """
    Train a model using the specified model class and parameters.

    Parameters:
    model_class (class): Model class to be used for training.
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target.
    param_grid (dict, optional): Parameter grid for GridSearchCV/ RandomizedSearchCV. Defaults to {}.
    best_combination (bool, optional): Whether to use GridSearchCV/RandomizedSearchCV for best parameter combination. Defaults to False.
    n_trials (int, optional): Number of trials for Optuna. Defaults to 10.
    **args: Additional arguments for the model class.

    Returns:
    model: Trained model.
    """

    if best_combination:
        model = model_class(**args)
        
        # Optuna
        def objective(trial):
            params = {key: trial.suggest_categorical(key, value) if isinstance(value, list) else trial.suggest_uniform(key, value[0], value[1]) for key, value in param_grid.items()}
            model.set_params(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_train)[:, 1]
            auc = roc_auc_score(y_train, y_pred)
            return auc

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        best_params_optuna = study.best_params
        best_model_optuna = model.set_params(**best_params_optuna)
        best_model_optuna.fit(X_train, y_train)

        # GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params_grid = grid_search.best_params_
        best_model_grid = grid_search.best_estimator_

        # RandomizedSearchCV
        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=3, n_iter=n_trials, n_jobs=-1)
        random_search.fit(X_train, y_train)
        best_params_random = random_search.best_params_
        best_model_random = random_search.best_estimator_

        # Compare models and select the best one
        models = {
            'Optuna': best_model_optuna,
            'GridSearchCV': best_model_grid,
            'RandomizedSearchCV': best_model_random,
        }

        best_model_name = None
        best_model_score = -float('inf')
        best_model = None

        for name, model in models.items():
            y_pred = model.predict_proba(X_train)[:, 1]
            auc = roc_auc_score(y_train, y_pred)
            if auc > best_model_score:
                best_model_score = auc
                best_model_name = name
                best_model = model

        print(f"Best model: {best_model_name} with AUC: {best_model_score}")

        return best_model

    else:
        model = model_class(**args)
        model.fit(X_train, y_train)
        model_to_save = model

    with open('trained_model.pickle', 'wb') as f:
        dill.dump(model_to_save, f)

    return model_to_save

with open('train_model.pickle', 'wb') as f:
    dill.dump(train_model, f)



# %%
def predict_model(df_test, model, features = []):

    '''
    Predict using the trained model on the test data.

    Parameters:
    df_test (DataFrame): The test DataFrame.
    model (model): The trained model.
    features (list): List of features to drop before prediction.

    Returns:
    array: The predicted probabilities.
    '''

    if type(model) == keras.src.models.sequential.Sequential:
        X_new = df_test.drop(columns=features)
        y_new_pred = model.predict(X_new)[:, 1]

    else:
        X_new = df_test.drop(columns=features)
        y_new_pred = model.predict_proba(X_new)[:, 1] 

    return y_new_pred


with open('predict_model.pickle', 'wb') as f:
    dill.dump(predict_model, f) 



"""# Build Neural Network Model"""

def neural_network_model(X, y, loss='binary_crossentropy', metrics='auc', activations='relu', output_activation='softmax', widths=[64], num_layers=0, epochs=50, batch_size=32, learning_rate=0.001, validation_split=0.3333):
    
    '''
    Build and train a neural network model.

    Parameters:
    X (DataFrame): The input features.
    y (Series): The target values.
    loss (str): The loss function.
    metrics (str): The evaluation metric.
    activations (str or list): The activation function(s) for hidden layers.
    output_activation (str): The activation function for the output layer.
    widths (list): The widths of the hidden layers.
    num_layers (int): The number of hidden layers.
    epochs (int): The number of training epochs.
    batch_size (int): The batch size for training.
    learning_rate (float): The learning rate for the optimizer.
    validation_split (float): The fraction of data to use for validation.

    Returns:
    tuple: The trained model and the training history.
    '''

    model_nn = Sequential()
    model_nn.add(Input((X.shape[1],)))
    
    if isinstance(activations, list):
        for i in range(num_layers):
            activation = activations[i % len(activations)]  # Rotate through the activations list
            width = widths[i % len(widths)]  # Rotate through the widths list
            model_nn.add(Dense(width, activation=activation))
            model_nn.add(Dropout(0.4))
    else:
        for i in range(num_layers):
            width = widths[i % len(widths)]  # Rotate through the widths list
            model_nn.add(Dense(width, activation=activations))
            model_nn.add(Dropout(0.4))
    
    model_nn.add(Dense(widths[-1], activation=output_activation))  # Output layer activation

     # Early Stopping callback
    es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=10)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model_nn.compile(loss=loss, optimizer=opt, metrics=[metrics])

   

    history = model_nn.fit(X, tf.keras.utils.to_categorical(y), epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[es])

    return model_nn, history