import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

@st.cache_data
def load_data():
    """
    Load training and target data from CSV files.
    """
    df = pd.read_csv('data/lucas_organic_carbon_training_and_test_data.csv')
    df_target = pd.read_csv('data/lucas_organic_carbon_target.csv')
    return df, df_target

@st.cache_data
def prepare_data(df, df_target):
    """
    Encode labels, standardize features, and split data into training and test sets.
    """
    le = LabelEncoder()
    df_target_encoded = le.fit_transform(df_target.values.ravel())
    sc_x = StandardScaler()
    X_standardized = pd.DataFrame(sc_x.fit_transform(df), columns=df.columns)
    data_train, data_test, target_train, target_test = train_test_split(
        X_standardized, df_target_encoded, test_size=0.25, random_state=42
    )
    return data_train, data_test, target_train, target_test, le
