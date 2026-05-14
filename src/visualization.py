import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import shap
import plotly.graph_objects as go

def plot_accuracy_bar_chart(accuracy, model_name):
    """
    Create and display an accuracy bar chart for a single model.
    """
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
    """
    Create and display an accuracy bar chart for comparing two models.
    """
    accuracy_df = pd.DataFrame({
        'Model': [model_name_m1, model_name_m2],
        'Accuracy': [accuracy_m1, accuracy_m2]
    })
    bar = alt.Chart(accuracy_df).mark_bar().encode(
        x=alt.X('Accuracy:Q', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('Model:N', sort='-x'),
        color=alt.Color('Model:N', legend=None)
    ).properties(
        width=250,
        height=150
    )
    st.altair_chart(bar, use_container_width=True)

def plot_acc_bar_chart_full_comp(accuracy_m1, accuracy_m2, accuracy_m3, model_name_m1, model_name_m2, model_name_m3):
    """
    Create and display an accuracy bar chart for comparing three models.
    """
    accuracy_df = pd.DataFrame({
        'Model': [model_name_m1, model_name_m2, model_name_m3],
        'Accuracy': [accuracy_m1, accuracy_m2, accuracy_m3]
    })
    bar = alt.Chart(accuracy_df).mark_bar().encode(
        x=alt.X('Accuracy:Q', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('Model:N', sort='-x'),
        color=alt.Color('Model:N', legend=None)
    ).properties(
        width=250,
        height=150
    )
    st.altair_chart(bar, use_container_width=True)

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Calculate and display a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    return fig

def generate_Shapley(X, model):
    """
    Generate Shapley values for the given model and data.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer

def generate_Sankey(predictions, actuals, le=None):
    """
    Generate a Sankey diagram to visualize prediction flow.
    """
    category_mapping_sankey = {0: 'high', 1: 'low', 2: 'moderate', 3: 'very high', 4: 'very low'}
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

    color_array_nodes = [0,1,2,3,4] * 2
    transition_counts = sankey_data.groupby(['Predicted', 'Actual']).size().reset_index(name='count')
    labels = list(set(transition_counts['Predicted']).union(set(transition_counts['Actual'])))
    label_map = {label: i for i, label in enumerate(labels)}
    sources = transition_counts['Predicted'].map(label_map).tolist()
    targets = (transition_counts['Actual'].map(label_map)+5).tolist()
    weights = transition_counts['count'].tolist()
    labels += labels
    color_array_link = list(map(color_dict.get, targets))
    color_array_nodes = list(map(color_dict.get, color_array_nodes))

    fig = go.Figure(data=[go.Sankey(
        node=dict(
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
    return fig

def generate_Sankey_three(cat_pred, xgb_pred, lgb_pred, actual):
    """
    Generate a Sankey diagram comparing three models against actual values.
    """
    prediction_1 = cat_pred[:]
    prediction_2 = xgb_pred[:]
    prediction_3 = lgb_pred[:]
    actuals = actual[:]
    
    category_mapping_sankey = {
        0: 'cat_high', 1: 'cat_low', 2: 'cat_moderate', 3: 'cat_very high', 4: 'cat_very low',
        5: 'xgb_high', 6: 'xgb_low', 7: 'xgb_moderate', 8: 'xgb_very high', 9: 'xgb_very low',
        10: 'lgb_high', 11: 'lgb_low', 12: 'lgb_moderate', 13: 'lgb_very high', 14: 'lgb_very low',
        15: 'actual_high', 16: 'actual_low', 17: 'actual_moderate', 18: 'actual_very high', 19: 'actual_very low'
    }

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

    transition_counts_1 = sankey_data_1.groupby(['Predicted', 'Actual']).size().reset_index(name='count')
    transition_counts_2 = sankey_data_2.groupby(['Predicted', 'Actual']).size().reset_index(name='count')
    transition_counts_3 = sankey_data_3.groupby(['Predicted', 'Actual']).size().reset_index(name='count')

    sources_1 = transition_counts_1['Predicted'].map({k: v for v, k in category_mapping_sankey.items()}).tolist()
    targets_1 = transition_counts_1['Actual'].map({k: v for v, k in category_mapping_sankey.items()}).tolist()
    weights_1 = transition_counts_1['count'].tolist()

    sources_2 = transition_counts_2['Predicted'].map({k: v for v, k in category_mapping_sankey.items()}).tolist()
    targets_2 = transition_counts_2['Actual'].map({k: v for v, k in category_mapping_sankey.items()}).tolist()
    weights_2 = transition_counts_2['count'].tolist()

    sources_3 = transition_counts_3['Predicted'].map({k: v for v, k in category_mapping_sankey.items()}).tolist()
    targets_3 = transition_counts_3['Actual'].map({k: v for v, k in category_mapping_sankey.items()}).tolist()
    weights_3 = transition_counts_3['count'].tolist()

    labels = ['cat_high', 'cat_low', 'cat_moderate', 'cat_very high', 'cat_very low', 
              'xgb_high', 'xgb_low', 'xgb_moderate', 'xgb_very high', 'xgb_very low', 
              'lgb_high', 'lgb_low', 'lgb_moderate', 'lgb_very high', 'lgb_very low', 
              'actual_high', 'actual_low', 'actual_moderate', 'actual_very high', 'actual_very low']
    unique = list(range(20))
    sources = sources_1 + sources_2 + sources_3
    targets = targets_1 + targets_2 + targets_3
    weights = weights_1 + weights_2 + weights_3

    color_array_link = list(map(color_dict.get, targets))
    color_array_nodes = list(map(color_dict.get, unique))

    fig = go.Figure(data=[go.Sankey(
        node=dict(
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
    return fig
