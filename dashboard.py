import xgboost as xgb
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import numpy as np
import shap
import lightgbm as lgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import os
import plotly.io as pio




# Daten laden
@st.cache_data
def load_data():
    df = pd.read_csv('lucas_organic_carbon_training_and_test_data.csv')
    df_target = pd.read_csv('lucas_organic_carbon_target.csv')
    return df, df_target

# Aufteilung der Daten in Trainings und Testsets
@st.cache_data
def prepare_data(df, df_target):
    le = LabelEncoder()
    df_target_encoded = le.fit_transform(df_target.values.ravel())
    sc_x = StandardScaler()
    X_standardized = pd.DataFrame(sc_x.fit_transform(df), columns=df.columns)
    data_train, data_test, target_train, target_test = train_test_split(X_standardized, df_target_encoded, test_size=0.25, random_state=42)
    return data_train, data_test, target_train, target_test, le


# -------------------Modeltraning--------------------
# XGBoost-Parameter
def set_params_xgb():
    params_xgb = {
        'device': 'cuda',
        'tree_method': 'hist',
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 5,
        'eval_metric': 'mlogloss'
    }
    return params_xgb

# Modelltraining XGBoost
def train_model(data_train, target_train, params_xgb):
    dtrain = xgb.DMatrix(data_train, label=target_train)
    bst = xgb.train(params_xgb, dtrain)
    return bst

# LightGBM-Parameter
def set_params_lgb():
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

# Modelltraining für LightGBM
#@st.cache_resource
def train_model_lgb(data_train, target_train, params_lgb):
    d_train = lgb.Dataset(data_train, label=target_train)
    clf = lgb.train(params_lgb, d_train, 100)
    return clf

#Catboost
# CatBoost-Parameter
def set_params_catboost():
    params_catboost = {
        'iterations': 100,
        'depth': 3,
        'learning_rate': 0.19,
        'loss_function': 'MultiClass',
        'verbose': True,
        'task_type': "CPU", #GPU wenn vorhanden
        'devices': '0:1'
    }
    return params_catboost

# Modelltraining für CatBoost
#@st.cache_resource
def train_model_catboost(data_train, target_train, params_catboost):
    cat_model = CatBoostClassifier(**params_catboost)
    cat_model.fit(data_train, target_train)
    return cat_model

def calculate_accuracy_and_predictions_general(model, data_test, target_test, model_type):
    if model_type == 'bst':
        dtest = xgb.DMatrix(data_test)
        y_pred = model.predict(dtest)
    elif model_type == 'clf':
        y_pred_raw = model.predict(data_test)
        y_pred = [np.argmax(line) for line in y_pred_raw]
    elif model_type == 'cat_model':
        y_pred = model.predict(data_test).flatten()
    else:
        raise ValueError(f"Unbekannter Modelltyp: {model_type}")

    accuracy = accuracy_score(target_test, y_pred)
    return accuracy, y_pred


# -------------------Grafiken plotten--------------------

# Funktion zum Erstellen und Anzeigen des Genauigkeitsbalkendiagramms
def plot_accuracy_bar_chart(accuracy, model_name):
    accuracy_df = pd.DataFrame({'Model': [model_name], 'Accuracy': [accuracy]})
    bar = alt.Chart(accuracy_df).mark_bar().encode(
        x=alt.X('Accuracy:Q', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('Model:N', sort='-x'),
        color=alt.value('green')
    ).properties(
        width=250,
        height=75
    )
    st.altair_chart(bar, theme=None, use_container_width=True)

def plot_acc_bar_chart_comp(accuracy_m1, accuracy_m2, model_name_m1, model_name_m2):
    # Erstellen Sie ein DataFrame für jede Genauigkeit
    accuracy_df = pd.DataFrame({
        'Model': [model_name_m1, model_name_m2],
        'Accuracy': [accuracy_m1, accuracy_m2]
    })

    # Erstellen Sie das Balkendiagramm
    bar = alt.Chart(accuracy_df).mark_bar().encode(
        x=alt.X('Accuracy:Q', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('Model:N', sort='-x'),
        color=alt.Color('Model:N', legend=None)
    ).properties(
        width=250,
        height=150
    )

    # Zeigen Sie das Diagramm an
    st.altair_chart(bar, use_container_width=True)

def plot_acc_bar_chart_full_comp(accuracy_m1, accuracy_m2, accuracy_m3, model_name_m1, model_name_m2, model_name_m3):
    # Erstellen Sie ein DataFrame für jede Genauigkeit
    accuracy_df = pd.DataFrame({
        'Model': [model_name_m1, model_name_m2, model_name_m3],
        'Accuracy': [accuracy_m1, accuracy_m2, accuracy_m3]
    })

    # Erstellen Sie das Balkendiagramm
    bar = alt.Chart(accuracy_df).mark_bar().encode(
        x=alt.X('Accuracy:Q', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('Model:N', sort='-x'),
        color=alt.Color('Model:N', legend=None)
    ).properties(
        width=250,
        height=150
    )

    # Zeigen Sie das Diagramm an
    st.altair_chart(bar, use_container_width=True)


# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    # Berechnung der Konfusionsmatrix
    cm = confusion_matrix(y_true, y_pred)
    # Erstellung der Konfusionsmatrix-Anzeige
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    # Erstellung des Plots
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    #plt.title('Confusion Matrix')
    plt.show()
    return fig


# Shapley Values
# Funktion zum Generieren der Shapley-Werte
def generate_Shapley(X, model):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer


# Sankey Diagram
def generate_Sankey(predictions, actuals, le=None):
    # Kategorie-Mapping für Sankey
    category_mapping_sankey = {0: 'high', 1: 'low', 2: 'moderate', 3: 'very high', 4: 'very low'}

    # Erstellen eines DataFrames mit den vorhergesagten und tatsächlichen Werten
    sankey_data = pd.DataFrame({
        'Predicted': [category_mapping_sankey[val] for val in predictions],
        'Actual': [category_mapping_sankey[val] for val in actuals]
    })
    color_dict = {0: 'rgba(202,65,94,0.7)', 1: 'rgba(205,162,10,0.7)',
                  2: 'rgba(55,78,25,0.7)', 3: 'rgba(52,90,100,0.7)',
                  4: 'rgba(25,120,90,0.7)',
                  5: 'rgba(202,65,94,0.7)', 6: 'rgba(205,162,10,0.7)',
                  7: 'rgba(55,78,25,0.7)', 8: 'rgba(52,90,100,0.7)',
                  9: 'rgba(25,120,90,0.7)'}

    color_array = ['rgba(202,65,94,0.7)', 'orange', 'green', 'yellow', 'red',
                   'rgba(202,65,94,0.7)', 'orange', 'green', 'red',
                   'rgba(202,65,94,0.7)', 'orange', 'green', 'yellow', 'red',
                   'rgba(202,65,94,0.7)', 'green', 'yellow',
                   'blue', 'orange', 'green', 'red']
    color_array_nodes = [0,1,2,3,4] * 2

    # Gruppieren der Daten und Zählen der Übergänge
    transition_counts = sankey_data.groupby(['Predicted', 'Actual']).size().reset_index(name='count')

    # Erstellen von Listen für die Knoten und Verbindungen des Sankey-Diagramms
    labels = list(set(transition_counts['Predicted']).union(set(transition_counts['Actual'])))
    label_map = {label: i for i, label in enumerate(labels)}
    sources = transition_counts['Predicted'].map(label_map).tolist()
    targets = (transition_counts['Actual'].map(label_map)+5).tolist()
    weights = transition_counts['count'].tolist()
    labels += labels
    color_array_link = list(map(color_dict.get, targets))
    color_array_nodes = list(map(color_dict.get, color_array_nodes))


    # Erstellen des Sankey-Diagramms
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            #valueformat=".0f",
            #valuesuffix="TWh",
            #pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=color_array_nodes,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=weights,
            color = color_array_link
        )
    )])


    # Rückgabe des Sankey-Diagramms
    return fig


def generate_Sankey_three(cat_pred, xgb_pred, lgb_pred, actual):
    print("sankey_start")
    prediction_1 = cat_pred[:]
    prediction_2 = xgb_pred[:]
    prediction_3 = lgb_pred[:]
    actuals = actual[:]
    # Kategorie-Mapping für Sankey
    category_mapping_sankey = {
        0: 'cat_high', 1: 'cat_low', 2: 'cat_moderate', 3: 'cat_very high', 4: 'cat_very low',
        5: 'xgb_high', 6: 'xgb_low', 7: 'xgb_moderate', 8: 'xgb_very high', 9: 'xgb_very low',
        10: 'lgb_high', 11: 'lgb_low', 12: 'lgb_moderate', 13: 'lgb_very high', 14: 'lgb_very low',
        15: 'actual_high', 16: 'actual_low', 17: 'actual_moderate', 18: 'actual_very high', 19: 'actual_very low'
        }
    # Erstellen eines DataFrames mit den vorhergesagten und tatsächlichen Werten
    sankey_data_1 = pd.DataFrame({
        'Predicted': [category_mapping_sankey[val] for val in prediction_1],
        'Actual': [category_mapping_sankey[val] for val in actuals]
    })

    sankey_data_2 = pd.DataFrame({
        'Predicted': [category_mapping_sankey[val] for val in prediction_2],
        'Actual': [category_mapping_sankey[val] for val in actuals]
    })
    sankey_data_3 = pd.DataFrame({
        'Predicted': [category_mapping_sankey[val] for val in prediction_3],
        'Actual': [category_mapping_sankey[val] for val in actuals]
    })
    color_dict = {0: 'rgba(202,10,10,0.7)', 1: 'rgba(205,192,10,0.7)',
                  2: 'rgba(254,10,225,0.7)', 3: 'rgba(10,190,150,0.7)',
                  4: 'rgba(12,10,220,0.7)',
                  5: 'rgba(202,10,10,0.7)', 6: 'rgba(205,192,10,0.7)',
                  7: 'rgba(254,10,225,0.7)', 8: 'rgba(10,190,150,0.7)',
                  9: 'rgba(12,10,220,0.7)',
                  10: 'rgba(202,10,10,0.7)', 11: 'rgba(205,192,10,0.7)',
                  12: 'rgba(254,10,225,0.7)', 13: 'rgba(10,190,150,0.7)',
                  14: 'rgba(12,10,220,0.7)',
                  15: 'rgba(202,10,10,0.7)', 16: 'rgba(205,192,10,0.7)',
                  17: 'rgba(254,10,225,0.7)', 18: 'rgba(10,190,150,0.7)',
                  19: 'rgba(12,10,220,0.7)'
                  }



    # Erstellen von Listen für die Knoten und Verbindungen des Sankey-Diagramms
    # Gruppieren der Daten und Zählen der Übergänge
    transition_counts_1 = sankey_data_1.groupby(['Predicted', 'Actual']).size().reset_index(name='count')
    transition_counts_2 = sankey_data_2.groupby(['Predicted', 'Actual']).size().reset_index(name='count')
    transition_counts_3 = sankey_data_3.groupby(['Predicted', 'Actual']).size().reset_index(name='count')

    # Erstellen von Listen für die Knoten und Verbindungen des Sankey-Diagramms
    labels_1 = list(set(transition_counts_1['Predicted']).union(set(transition_counts_1['Actual'])))
    label_map_1 = {label: i for i, label in enumerate(labels_1)}
    sources_1 = transition_counts_1['Predicted'].map(label_map_1).tolist()
    targets_1 = (transition_counts_1['Actual'].map(label_map_1) + 15).tolist()
    weights_1 = transition_counts_1['count'].tolist()

    labels_2 = list(set(transition_counts_2['Predicted']).union(set(transition_counts_2['Actual'])))
    label_map_2 = {label: i for i, label in enumerate(labels_2)}
    sources_2 = (transition_counts_2['Predicted'].map(label_map_2) + 5).tolist()
    targets_2 = (transition_counts_2['Actual'].map(label_map_2) + 15).tolist()
    weights_2 = transition_counts_2['count'].tolist()

    labels_3 = list(set(transition_counts_3['Predicted']).union(set(transition_counts_3['Actual'])))
    label_map_3 = {label: i for i, label in enumerate(labels_3)}
    sources_3 = (transition_counts_3['Predicted'].map(label_map_3) + 10).tolist()
    targets_3 = (transition_counts_3['Actual'].map(label_map_3) + 15).tolist()
    weights_3 = transition_counts_3['count'].tolist()

    labels = ['cat_high', 'cat_low', 'cat_moderate', 'cat_very high', 'cat_very low', 'xgb_high', 'xgb_low', 'xgb_moderate', 'xgb_very high', 'xgb_very low', 'lgb_high', 'lgb_low', 'lgb_moderate', 'lgb_very high', 'lgb_very low', 'actual_high', 'actual_low', 'actual_moderate', 'actual_very high', 'actual_very low']
    unique = [ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19]
    sources = sources_1 + sources_2 + sources_3
    targets = targets_1 + targets_2 + targets_3
    weights = weights_1 + weights_2 + weights_3

    color_array_link = list(map(color_dict.get, targets))
    color_array_nodes = list(map(color_dict.get, unique))

    # Erstellen des Sankey-Diagramms
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            #valueformat=".0f",
            #valuesuffix="TWh",
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=color_array_nodes,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=weights,
            color=color_array_link
        )
    )])

    print("sankey return")
    # Rückgabe des Sankey-Diagramms
    return fig

# Define a function to write log messages
def log_message(message):
    print(message)

# Function to get the modification timestamp of a file
def get_file_modification_timestamp(file_path):
    if os.path.exists(file_path):
        return os.path.getmtime(file_path)
    else:
        log_message("Keine Datei vorhanden")
        return 0.0

# Function to save an image to a specified folder
def save_image_to_folder(fig, folder_path, image_name):
    image_file_path = os.path.join(folder_path, image_name)

    # Check if fig is a Plotly figure
    if isinstance(fig, go.Figure):
        pio.write_image(fig, image_file_path)
    else:
        # Assuming fig is a SHAP summary plot, handled via Matplotlib
        # Note: This assumes you've already created the plot with SHAP and it's ready to be saved
        fig.savefig(image_file_path, bbox_inches='tight')


def checkFoldersAndSaveNewImage(fig, file_name):
    image_file_path1 = os.path.join('folder1', file_name)
    timestamp1 = get_file_modification_timestamp(image_file_path1)
    log_message("Timestamp1: " + str(timestamp1))
    image_file_path2 = os.path.join('folder2', file_name)
    timestamp2 = get_file_modification_timestamp(image_file_path2)
    log_message("Timestamp 2: " + str(timestamp2))

    if timestamp1 == 0.0:
        save_image_to_folder(fig, 'folder1', file_name)
        log_message("In folder1 gespeichert, da hier nichts drin ist")
        return

    if timestamp2 == 0.0:
        save_image_to_folder(fig, 'folder2', file_name)
        log_message("In folder2 gespeichert, da hier nichts drin ist")
        return

    if(timestamp1-timestamp2 < 0):
        save_image_to_folder(fig, 'folder1', file_name)
        log_message("In folder1 gespeichert")
    elif(timestamp1-timestamp2 > 0):
        save_image_to_folder(fig, 'folder2', file_name)
        log_message("In folder2 gespeichert")
    else:
        save_image_to_folder(fig, 'folder1', file_name)
        log_message("In keine der beide IFs rein deswegen folder1")

def createSaveButton(fig, file_name):
    # Create a button in the sidebar
    if st.sidebar.button(file_name + ' speichern'):
        # What to do when the button is clicked
        checkFoldersAndSaveNewImage(fig, file_name)
        log_message("Ende______________________________________________________________________________________")
        st.write('Diagramm gespeichert!')


def compareCharts(file_name):
    final_image_file_path1 = 'RIP'
    final_image_file_path2 = 'RIP'

    image_file_path1 = os.path.join('folder1', file_name)
    image_file_path2 = os.path.join('folder2', file_name)
    timestamp1 = get_file_modification_timestamp(image_file_path1)
    timestamp2 = get_file_modification_timestamp(image_file_path2)

    if timestamp1 < timestamp2:
        final_image_file_path1 = image_file_path1
        final_image_file_path2 = image_file_path2
    else:
        final_image_file_path1 = image_file_path2
        final_image_file_path2 = image_file_path1

    col1, col2 = st.columns(2)
    with col1:
        st.header('***Vorher:***')
        st.image(final_image_file_path1, use_column_width="always")
    with col2:
        st.header('***Nachher:***')
        st.image(final_image_file_path2, use_column_width="always")




# -------------------Streamlit Dashboard--------------------
def main():


    st.set_page_config(layout="wide")

    st.title('ModelWise 23/24')

    df, df_target = load_data()
    #data_train, data_test, target_train, target_test, le = prepare_data(df, df_target)
    array = np.array([0, 1, 2, 3, 4])
    category_mapping_sankey= {0:'high', 1:'low', 2:'moderate', 3:'very high', 4:'very low'}
    colors = {'high': 'orange', 'low': 'green', 'moderate': 'yellow', 'very high': 'red', 'very low': 'blue'}
    class_names = ['high', 'low', 'moderate', 'very high', 'very low']

    # -------------------Datensatz auswählen--------------------
    if df is not None:
        data = df
        st.sidebar.write('Datensatz anpassen:')

        all_columns = data.columns.tolist()
        if st.sidebar.checkbox('Select All'):
            selected_columns = all_columns[0:]
        else:
            selected_columns = st.sidebar.multiselect('Select columns', all_columns, all_columns[0])
            #st.dataframe(data[selected_columns])
        df = data[selected_columns]

        # Hinzufügen des Sliders zur Auswahl des Prozentsatzes des Datensatzes
        #percentage = st.sidebar.slider('Wähle den Prozentsatz des Datensatzes:', min_value=10, max_value=100, value=70)
        #df = df.sample(frac=percentage / 100.0)

        if st.sidebar.checkbox('Modelle trainieren'):

            # Daten aufteilen
            data_train, data_test, target_train, target_test, le = prepare_data(df, df_target)
            st.sidebar.write('Datensatz ausgewählt')

    for folder_name in ['folder1', 'folder2']:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            log_message(f"Ordner {folder_name} erstellt.")

    # -------------------Modell auswählen--------------------
    model_option = st.sidebar.selectbox('Wähle ein Modell für die Analyse:', ['Wähle das Modell','XGBoost', 'LightGBM', 'CatBoost','XGBoost vs LightGBM', 'XGBoost vs Catboost', 'LightGBM vs Catboost', 'Alle Modelle'])

    warnings.filterwarnings('ignore')

    #array = np.array([0, 1, 2, 3, 4])
    #category_mapping = le.inverse_transform(array)
    #classnames = le.inverse_transform(array)
    class_names_s = {0: 'high', 1: 'low', 2: 'moderate', 3: 'very high', 4: 'very low'}
    class_names_lgb = {0: 'very low', 1: 'very high', 2: 'high', 3: 'moderate', 4: 'low'}
    class_names_cat = {0: 'high', 1: 'very high', 2: 'low', 3: 'moderate', 4: 'very low'}
    classnames_1 = list(class_names_s.values())
    classnames_lgb = list(class_names_lgb.values())
    classnames_cat = list(class_names_cat.values())
    #print("--------------------------------------------------------------------------------------------------------------------------------------------")
    #print(classnames_1)

    # -------------------einzelnes Modell--------------------
    if model_option == 'Wähle das Modell':
        pass

    # XGBoost
    elif model_option == 'XGBoost':
        try:
            #Modeltraining

            params = set_params_xgb()
            bst = train_model(data_train, target_train, params)

            # Accuracy berechnen
            accuracy, xgboost_pred = calculate_accuracy_and_predictions_general(bst, data_test, target_test, 'bst')
            st.header(f'***Genauigkeit von XGBoost: {accuracy:.2f}***')

            plot_accuracy_bar_chart(accuracy, 'XGBoost')

            if st.sidebar.checkbox('Sankey Diagram erstellen'):

                st.header('***Sankey Diagram für XGBoost:***')
                sankey_fig = generate_Sankey(xgboost_pred, target_test, le)
                st.plotly_chart(sankey_fig, use_container_width=True)

                createSaveButton(sankey_fig, 'sankey_chart.png')

            if st.sidebar.checkbox('Shapley Values berechnen'):
                st.subheader('***Shapley-Werte für XGBoost:***')

                shap_values, explainer = generate_Shapley(data_train, bst)
                st.set_option('deprecation.showPyplotGlobalUse', False)

                # Größe des Plots festlegen
                #plt.figure(figsize=(5, 3))  # Passen Sie die Breite und Höhe nach Bedarf an


                shap.summary_plot(shap_values, data_train, plot_type="bar", class_names=classnames_1, show=False)
                #st.pyplot(bbox_inches='tight', pad_inches=0)


                fig = plt.gcf()
                # Create a button in the sidebar
                createSaveButton(fig, 'shapley_chart.png')

                st.pyplot(bbox_inches='tight', pad_inches=0)

            if st.sidebar.checkbox('Confusion Matrix generieren'):

                st.subheader('***Confusionmatrix für XGBoost:***')
                confusion_fig = plot_confusion_matrix(target_test, xgboost_pred, class_names)
                #st.pyplot(confusion_fig)

                createSaveButton(confusion_fig, 'matrix_chart.png')

                st.pyplot(confusion_fig)
            st.write('XGBoost-Modell wurde ausgewählt.')

        except:
            st.write('Bitte trainiere dein Modell!')


    #LightGBM
    elif model_option == 'LightGBM':
        try:
            # LightGBM-Modell
            params_lgb = set_params_lgb()
            clf = train_model_lgb(data_train, target_train, params_lgb)

            accuracy, lightgbm_pred = calculate_accuracy_and_predictions_general(clf, data_test, target_test, 'clf')
            st.header(f'***Genauigkeit von LightGBM: {accuracy:.2f}***')

            plot_accuracy_bar_chart(accuracy, 'LightGBM')

            if st.sidebar.checkbox('Sankey Diagram erstellen'):
                # Generiere das Sankey-Diagramm für LightGBM
                st.header('***Sankey Diagram für LightGBM:***')
                sankey_fig = generate_Sankey(lightgbm_pred, target_test, le)
                st.plotly_chart(generate_Sankey(lightgbm_pred, target_test, le), use_container_width=True)

                createSaveButton(sankey_fig, 'sankey_chart.png')

            if st.sidebar.checkbox('Shapley Values berechnen'):

                # Shapley Values
                shap_values, explainer = generate_Shapley(data_test, clf)
                st.header('***Shapley-Werte für LightGBM:***')
                shap_values, explainer = generate_Shapley(data_train, clf)


                shap.summary_plot(shap_values, data_train, plot_type="bar",class_names=classnames_lgb, show=False)


                fig = plt.gcf()
                # Create a button in the sidebar
                createSaveButton(fig, 'shapley_chart.png')

                st.pyplot(bbox_inches='tight', pad_inches=0)

            if st.sidebar.checkbox('Confusion Matrix generieren'):

                # Berechnung der Confusion Matrix
                fig = plot_confusion_matrix(target_test, lightgbm_pred, class_names)
                st.pyplot(fig)

                createSaveButton(fig, 'matrix_chart.png')

            st.write('LightGBM-Modell wurde ausgewählt.')

        except:
            st.write('Bitte trainiere dein Modell!')


    # CatBoost
    elif model_option == 'CatBoost':
        try:
            params_catboost = set_params_catboost()
            cat_model = train_model_catboost(data_train, target_train, params_catboost)


            accuracy, catboost_pred = calculate_accuracy_and_predictions_general(cat_model, data_test, target_test, 'cat_model')
            st.header(f'***Genauigkeit von CatBoost: {accuracy:.2f}***')

            plot_accuracy_bar_chart(accuracy, 'CatBoost')

            if st.sidebar.checkbox('Sankey Diagram erstellen'):
                # Generiere das Sankey-Diagramm für CatBoost
                st.header('***Sankey Diagram für CatBoost:***')
                fig = generate_Sankey(catboost_pred, target_test, le)
                st.plotly_chart(fig, use_container_width=True)

                # Create a button in the sidebar
                createSaveButton(fig, 'sankey_chart.png')

            if st.sidebar.checkbox('Shapley Values berechnen'):
                #warnings.filterwarnings('ignore')
                # Shapley-Werte generieren
                shap_values, explainer = generate_Shapley(data_train, cat_model)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.header('***Shapley-Werte für CatBoost:***')
                shap.summary_plot(shap_values, data_train, plot_type="bar", class_names=classnames_cat,show=False)


                fig = plt.gcf()
                # Create a button in the sidebar
                createSaveButton(fig, 'shapley_chart.png')
                st.pyplot(bbox_inches='tight', pad_inches=0)

            if st.sidebar.checkbox('Confusion Matrix generieren'):

                # Berechnung der Confusion Matrix
                fig = plot_confusion_matrix(target_test, catboost_pred, class_names)
                st.pyplot(fig)

                createSaveButton(fig, 'matrix_chart.png')

            st.write('CatBoost-Modell wurde ausgewählt.')

        except:
            st.write('Bitte trainiere dein Modell!')



    # -------------------Modelle vergleichen--------------------
    elif model_option == 'XGBoost vs LightGBM':
        try:
            # XGBoost Parameter
            params = set_params_xgb()
            bst = train_model(data_train, target_train, params)

            # LightGBM
            params_lgb = set_params_lgb()
            clf = train_model_lgb(data_train, target_train, params_lgb)

            accuracy_xgb, xgboost_pred = calculate_accuracy_and_predictions_general(bst, data_test, target_test,'bst')
            accuracy_lgb, lightgbm_pred = calculate_accuracy_and_predictions_general(clf, data_test, target_test, 'clf')

            plot_acc_bar_chart_comp(accuracy_xgb, accuracy_lgb, 'XGBoost', 'LightGBM')
            col1, col2 = st.columns(2)

            if st.sidebar.checkbox('Sankey Diagram erstellen'):
                with col1:
                    st.header('***Sankey Diagram für XGBoost:***')
                    sankey_fig = generate_Sankey(xgboost_pred, target_test, le)
                    st.plotly_chart(sankey_fig, use_container_width=True)
                with col2:
                    st.header('***Sankey Diagram für LightGBM:***')
                    sankey_fig = generate_Sankey(lightgbm_pred, target_test, le)
                    st.plotly_chart(sankey_fig, use_container_width=True)

            if st.sidebar.checkbox('Shapley Values berechnen'):
                with col1:
                    st.subheader('***Shapley-Werte für XGBoost:***')
                    shap_values, explainer = generate_Shapley(data_train, bst)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    shap.summary_plot(shap_values, data_train, plot_type="bar",class_names=classnames_1, show=False)
                    st.pyplot(bbox_inches='tight', pad_inches=0)
                with col2:
                    st.subheader('***Shapley-Werte für LightGBM:***')
                    shap_values, explainer = generate_Shapley(data_train, clf)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    shap.summary_plot(shap_values, data_train, plot_type="bar",class_names=classnames_lgb, show=False)
                    st.pyplot(bbox_inches='tight', pad_inches=0)


            if st.sidebar.checkbox('Confusion Matrix generieren'):
                fig_xg = plot_confusion_matrix(target_test, xgboost_pred, class_names)
                fig_lgb = plot_confusion_matrix(target_test, lightgbm_pred, class_names)
                with col1:
                    st.subheader('***Confusionmatrix für XGBoost:***')
                    st.pyplot(fig_xg)
                with col2:
                    st.subheader('***Confusionmatrix für LightGBM:***')
                    st.pyplot(fig_lgb)

        except:
            st.write('Bitte trainiere deine Modelle!')

    elif model_option == 'XGBoost vs Catboost':
        try:
            # XGBoost Parameter
            params = set_params_xgb()
            bst = train_model(data_train, target_train, params)

            # Catboost
            params_catboost = set_params_catboost()
            cat_model = train_model_catboost(data_train, target_train, params_catboost)

            accuracy_xgb, xgboost_pred = calculate_accuracy_and_predictions_general(bst, data_test, target_test,'bst')
            accuracy_cat, catboost_pred = calculate_accuracy_and_predictions_general(cat_model, data_test, target_test, 'cat_model')

            plot_acc_bar_chart_comp(accuracy_xgb, accuracy_cat, 'XGBoost', 'Catboost')

            col1, col2 = st.columns(2)

            if st.sidebar.checkbox('Sankey Diagram erstellen'):
                with col1:
                    st.header('***Sankey Diagram für XGBoost:***')
                    sankey_fig = generate_Sankey(xgboost_pred, target_test, le)
                    st.plotly_chart(sankey_fig, use_container_width=True)
                with col2:
                    st.header('***Sankey Diagram für CatBoost:***')
                    sankey_fig = generate_Sankey(catboost_pred, target_test, le)
                    st.plotly_chart(sankey_fig, use_container_width=True)

            if st.sidebar.checkbox('Shapley Values berechnen'):
                with col1:
                    st.subheader('***Shapley-Werte für XGBoost:***')
                    shap_values, explainer = generate_Shapley(data_train, bst)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    shap.summary_plot(shap_values, data_train, plot_type="bar", class_names=classnames_1,show=False)
                    st.pyplot(bbox_inches='tight', pad_inches=0)
                with col2:
                    st.subheader('***Shapley-Werte für Catboost:***')
                    shap_values, explainer = generate_Shapley(data_train, cat_model)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    shap.summary_plot(shap_values, data_train, plot_type="bar",class_names=classnames_cat, show=False)
                    st.pyplot(bbox_inches='tight', pad_inches=0)

            if st.sidebar.checkbox('Confusion Matrix generieren'):
                fig_xg = plot_confusion_matrix(target_test, xgboost_pred, class_names)
                fig_cat = plot_confusion_matrix(target_test, catboost_pred, class_names)
                with col1:
                    st.subheader('***Confusionmatrix für XGBoost:***')
                    st.pyplot(fig_xg)
                with col2:
                    st.subheader('***Confusionmatrix für CatBoost:***')
                    st.pyplot(fig_cat)

        except:
            st.write('Bitte trainiere deine Modelle!')

    elif model_option == 'LightGBM vs Catboost':
        try:
            # LightGBM
            params_lgb = set_params_lgb()
            clf = train_model_lgb(data_train, target_train, params_lgb)

            # Catboost
            params_catboost = set_params_catboost()
            cat_model = train_model_catboost(data_train, target_train, params_catboost)

            accuracy_lgb, lightgbm_pred = calculate_accuracy_and_predictions_general(clf, data_test, target_test, 'clf')
            accuracy_cat, catboost_pred = calculate_accuracy_and_predictions_general(cat_model, data_test, target_test, 'cat_model')

            plot_acc_bar_chart_comp(accuracy_lgb, accuracy_cat, 'LightGBM', 'Catboost')

            col1, col2 = st.columns(2)

            if st.sidebar.checkbox('Sankey Diagram erstellen'):
                with col1:
                    st.header('***Sankey Diagram für LightGBM:***')
                    sankey_fig = generate_Sankey(lightgbm_pred, target_test, le)
                    st.plotly_chart(sankey_fig, use_container_width=True)
                with col2:
                    st.header('***Sankey Diagram für CatBoost:***')
                    sankey_fig = generate_Sankey(catboost_pred, target_test, le)
                    st.plotly_chart(sankey_fig, use_container_width=True)

            if st.sidebar.checkbox('Shapley Values berechnen'):
                with col1:
                    st.subheader('***Shapley-Werte für LightGBM:***')
                    shap_values, explainer = generate_Shapley(data_train, clf)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    shap.summary_plot(shap_values, data_train, plot_type="bar", class_names=classnames_lgb,show=False)
                    st.pyplot(bbox_inches='tight', pad_inches=0)
                with col2:
                    st.subheader('***Shapley-Werte für Catboost:***')
                    shap_values, explainer = generate_Shapley(data_train, cat_model)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    shap.summary_plot(shap_values, data_train, plot_type="bar",class_names=classnames_cat, show=False)
                    st.pyplot(bbox_inches='tight', pad_inches=0)

            if st.sidebar.checkbox('Confusion Matrix generieren'):
                fig_lgb = plot_confusion_matrix(target_test, lightgbm_pred, class_names)
                fig_cat = plot_confusion_matrix(target_test, catboost_pred, class_names)
                with col1:
                    st.subheader('***Confusionmatrix für LightGBM:***')
                    st.pyplot(fig_lgb)
                with col2:
                    st.subheader('***Confusionmatrix für CatBoost:***')
                    st.pyplot(fig_cat)

        except:
            st.write('Bitte trainiere deine Modelle!')

    elif model_option == 'Alle Modelle':
        try:
            # XGBoost Parameter
            params = set_params_xgb()
            bst = train_model(data_train, target_train, params)

            # LightGBM
            params_lgb = set_params_lgb()
            clf = train_model_lgb(data_train, target_train, params_lgb)

            # Catboost
            params_catboost = set_params_catboost()
            cat_model = train_model_catboost(data_train, target_train, params_catboost)

            accuracy_xgb, xgboost_pred = calculate_accuracy_and_predictions_general(bst, data_test, target_test, 'bst')
            accuracy_lgb, lightgbm_pred = calculate_accuracy_and_predictions_general(clf, data_test, target_test, 'clf')
            accuracy_cat, catboost_pred = calculate_accuracy_and_predictions_general(cat_model, data_test,target_test, 'cat_model')

            plot_acc_bar_chart_full_comp(accuracy_xgb ,accuracy_lgb, accuracy_cat, 'XGBoost','LightGBM', 'Catboost')
            #col1, col2, col3 = st.columns(3)

            if st.sidebar.checkbox('Sankey Diagram erstellen'):

                st.header('***Sankey Diagram Vergleich:***')
                sankey_fig = generate_Sankey_three(catboost_pred, xgboost_pred, lightgbm_pred, target_test)
                st.plotly_chart(sankey_fig, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            if st.sidebar.checkbox('Shapley Values berechnen'):
                with col1:
                    st.subheader('***Shapley-Werte für XGBoost:***')
                    shap_values, explainer = generate_Shapley(data_train, bst)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    shap.summary_plot(shap_values, data_train, plot_type="bar", class_names=classnames_1,show=False)
                    st.pyplot(bbox_inches='tight', pad_inches=0)
                with col2:
                    st.subheader('***Shapley-Werte für LightGBM:***')
                    shap_values, explainer = generate_Shapley(data_train, clf)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    shap.summary_plot(shap_values, data_train, plot_type="bar", class_names=classnames_lgb,show=False)
                    st.pyplot(bbox_inches='tight', pad_inches=0)
                with col3:
                    st.subheader('***Shapley-Werte für Catboost:***')
                    shap_values, explainer = generate_Shapley(data_train, cat_model)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    shap.summary_plot(shap_values, data_train, plot_type="bar", class_names=classnames_cat,show=False)
                    st.pyplot(bbox_inches='tight', pad_inches=0)


            if st.sidebar.checkbox('Confusion Matrix generieren'):
                fig_xg = plot_confusion_matrix(target_test, xgboost_pred, class_names)
                fig_lgb = plot_confusion_matrix(target_test, lightgbm_pred, class_names)
                fig_cat = plot_confusion_matrix(target_test, catboost_pred, class_names)
                with col1:
                    st.subheader('***Confusionmatrix für XGBoost:***')
                    st.pyplot(fig_xg)
                with col2:
                    st.subheader('***Confusionmatrix für LightGBM:***')
                    st.pyplot(fig_lgb)
                with col3:
                    st.subheader('***Confusionmatrix für Catboost:***')
                    st.pyplot(fig_cat)

        except:
            st.write('Bitte trainiere deine Modelle!')

    model_option2 = st.sidebar.selectbox('Wähle ein Diagramm für den Vergleich:', ['Wähle das Diagramm', 'Sankey', 'Shapley', 'Matrix'])

    if model_option2 == 'Sankey':
        compareCharts('sankey_chart.png')
    if model_option2 == 'Shapley':
        compareCharts('shapley_chart.png')
    if model_option2 == 'Matrix':
        compareCharts('matrix_chart.png')


if __name__ == "__main__":
    main()
