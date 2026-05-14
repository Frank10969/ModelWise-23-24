import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import numpy as np
from sklearn.metrics import accuracy_score

def set_params_xgb():
    """
    Set parameters for XGBoost model.
    """
    params_xgb = {
        'device': 'cuda',
        'tree_method': 'hist',
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 5,
        'eval_metric': 'mlogloss'
    }
    return params_xgb

def train_model(data_train, target_train, params_xgb):
    """
    Train XGBoost model.
    """
    dtrain = xgb.DMatrix(data_train, label=target_train)
    bst = xgb.train(params_xgb, dtrain)
    return bst

def set_params_lgb():
    """
    Set parameters for LightGBM model.
    """
    params_lgb = {
        'learning_rate': 0.19,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'max_depth': 3,
        'num_leaves': 12,
        'num_class': 5
    }
    return params_lgb

def train_model_lgb(data_train, target_train, params_lgb):
    """
    Train LightGBM model.
    """
    d_train = lgb.Dataset(data_train, label=target_train)
    clf = lgb.train(params_lgb, d_train, 100)
    return clf

def set_params_catboost():
    """
    Set parameters for CatBoost model.
    """
    params_catboost = {
        'iterations': 100,
        'depth': 3,
        'learning_rate': 0.19,
        'loss_function': 'MultiClass',
        'verbose': True,
        'task_type': "GPU", # Use GPU if available
        'devices': '0:1'
    }
    return params_catboost

def train_model_catboost(data_train, target_train, params_catboost):
    """
    Train CatBoost model.
    """
    cat_model = CatBoostClassifier(**params_catboost)
    cat_model.fit(data_train, target_train)
    return cat_model

def calculate_accuracy_and_predictions_general(model, data_test, target_test, model_type):
    """
    Calculate accuracy and get predictions based on model type.
    """
    if model_type == 'bst':
        dtest = xgb.DMatrix(data_test)
        y_pred = model.predict(dtest)
    elif model_type == 'clf':
        y_pred_raw = model.predict(data_test)
        y_pred = [np.argmax(line) for line in y_pred_raw]
    elif model_type == 'cat_model':
        y_pred = model.predict(data_test).flatten()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    accuracy = accuracy_score(target_test, y_pred)
    return accuracy, y_pred
