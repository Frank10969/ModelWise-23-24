import streamlit as st
import matplotlib.pyplot as plt
import shap
from src.models import (
    set_params_xgb, train_model, 
    set_params_lgb, train_model_lgb, 
    set_params_catboost, train_model_catboost,
    calculate_accuracy_and_predictions_general
)
from src.visualization import (
    plot_accuracy_bar_chart, plot_confusion_matrix, generate_Shapley, generate_Sankey
)
from src.utils import createSaveButton

def render_single_model_view(prepared_data, model_type):
    """
    Render analysis for a single model (XGBoost, LightGBM, or CatBoost).
    """
    data_train, data_test, target_train, target_test, le = prepared_data
    
    # Mappings
    class_names = ['high', 'low', 'moderate', 'very high', 'very low']
    class_names_s = {0: 'high', 1: 'low', 2: 'moderate', 3: 'very high', 4: 'very low'}
    class_names_lgb = {0: 'very low', 1: 'very high', 2: 'high', 3: 'moderate', 4: 'low'}
    class_names_cat = {0: 'very low', 1: 'low', 2: 'high', 3: 'moderate', 4: 'very high'}
    
    try:
        if model_type == 'XGBoost':
            params = set_params_xgb()
            model = train_model(data_train, target_train, params)
            acc_type = 'bst'
            display_names = list(class_names_s.values())
        elif model_type == 'LightGBM':
            params = set_params_lgb()
            model = train_model_lgb(data_train, target_train, params)
            acc_type = 'clf'
            display_names = list(class_names_lgb.values())
        elif model_type == 'CatBoost':
            params = set_params_catboost()
            model = train_model_catboost(data_train, target_train, params)
            acc_type = 'cat_model'
            display_names = list(class_names_cat.values())
        else:
            return

        # Calculate accuracy
        accuracy, predictions = calculate_accuracy_and_predictions_general(model, data_test, target_test, acc_type)
        st.header(f'***{model_type} Accuracy: {accuracy:.2f}***')
        plot_accuracy_bar_chart(accuracy, model_type)

        # Visualizations
        if st.sidebar.checkbox('Create Sankey Diagram'):
            st.header(f'***Sankey Diagram for {model_type}:***')
            sankey_fig = generate_Sankey(predictions, target_test, le)
            st.plotly_chart(sankey_fig, use_container_width=True)
            createSaveButton(sankey_fig, 'sankey_chart.png')

        if st.sidebar.checkbox('Calculate Shapley Values'):
            st.subheader(f'***Shapley Values for {model_type}:***')
            shap_values, _ = generate_Shapley(data_train, model)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.summary_plot(shap_values, data_train, plot_type="bar", class_names=display_names, show=False)
            fig = plt.gcf()
            createSaveButton(fig, 'shapley_chart.png')
            st.pyplot(bbox_inches='tight', pad_inches=0)

        if st.sidebar.checkbox('Generate Confusion Matrix'):
            st.subheader(f'***Confusion Matrix for {model_type}:***')
            confusion_fig = plot_confusion_matrix(target_test, predictions, class_names)
            createSaveButton(confusion_fig, 'matrix_chart.png')
            st.pyplot(confusion_fig)
        
        st.write(f'{model_type} model selected.')

    except Exception as e:
        st.write(f"An error occurred during {model_type} analysis: {e}")
