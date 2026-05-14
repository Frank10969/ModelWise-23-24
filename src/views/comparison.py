import streamlit as st
from src.models import (
    set_params_xgb, train_model, 
    set_params_lgb, train_model_lgb, 
    set_params_catboost, train_model_catboost,
    calculate_accuracy_and_predictions_general
)
from src.visualization import (
    plot_acc_bar_chart_comp, plot_acc_bar_chart_full_comp,
    plot_confusion_matrix, generate_Shapley, generate_Sankey, generate_Sankey_three
)
import shap

def render_comparison_view(prepared_data, comparison_type):
    """
    Render comparison between two models.
    """
    data_train, data_test, target_train, target_test, le = prepared_data
    class_names = ['high', 'low', 'moderate', 'very high', 'very low']
    
    # Name mappings for SHAP
    class_names_s = {0: 'high', 1: 'low', 2: 'moderate', 3: 'very high', 4: 'very low'}
    class_names_lgb = {0: 'very low', 1: 'very high', 2: 'high', 3: 'moderate', 4: 'low'}
    class_names_cat = {0: 'very low', 1: 'low', 2: 'high', 3: 'moderate', 4: 'very high'}

    try:
        if comparison_type == 'XGBoost vs LightGBM':
            m1, m1_name, m1_type, m1_params = train_model(data_train, target_train, set_params_xgb()), 'XGBoost', 'bst', class_names_s
            m2, m2_name, m2_type, m2_params = train_model_lgb(data_train, target_train, set_params_lgb()), 'LightGBM', 'clf', class_names_lgb
        elif comparison_type == 'XGBoost vs Catboost':
            m1, m1_name, m1_type, m1_params = train_model(data_train, target_train, set_params_xgb()), 'XGBoost', 'bst', class_names_s
            m2, m2_name, m2_type, m2_params = train_model_catboost(data_train, target_train, set_params_catboost()), 'Catboost', 'cat_model', class_names_cat
        elif comparison_type == 'LightGBM vs Catboost':
            m1, m1_name, m1_type, m1_params = train_model_lgb(data_train, target_train, set_params_lgb()), 'LightGBM', 'clf', class_names_lgb
            m2, m2_name, m2_type, m2_params = train_model_catboost(data_train, target_train, set_params_catboost()), 'Catboost', 'cat_model', class_names_cat
        else:
            return

        acc1, pred1 = calculate_accuracy_and_predictions_general(m1, data_test, target_test, m1_type)
        acc2, pred2 = calculate_accuracy_and_predictions_general(m2, data_test, target_test, m2_type)

        plot_acc_bar_chart_comp(acc1, acc2, m1_name, m2_name)
        col1, col2 = st.columns(2)

        if st.sidebar.checkbox('Create Sankey Diagram'):
            with col1:
                st.header(f'***Sankey Diagram for {m1_name}:***')
                st.plotly_chart(generate_Sankey(pred1, target_test, le), use_container_width=True)
            with col2:
                st.header(f'***Sankey Diagram for {m2_name}:***')
                st.plotly_chart(generate_Sankey(pred2, target_test, le), use_container_width=True)

        if st.sidebar.checkbox('Calculate Shapley Values'):
            with col1:
                st.subheader(f'***Shapley Values for {m1_name}:***')
                sv1, _ = generate_Shapley(data_train, m1)
                shap.summary_plot(sv1, data_train, plot_type="bar", class_names=list(m1_params.values()), show=False)
                st.pyplot(bbox_inches='tight', pad_inches=0)
            with col2:
                st.subheader(f'***Shapley Values for {m2_name}:***')
                sv2, _ = generate_Shapley(data_train, m2)
                shap.summary_plot(sv2, data_train, plot_type="bar", class_names=list(m2_params.values()), show=False)
                st.pyplot(bbox_inches='tight', pad_inches=0)

        if st.sidebar.checkbox('Generate Confusion Matrix'):
            with col1:
                st.subheader(f'***Confusion Matrix for {m1_name}:***')
                st.pyplot(plot_confusion_matrix(target_test, pred1, class_names))
            with col2:
                st.subheader(f'***Confusion Matrix for {m2_name}:***')
                st.pyplot(plot_confusion_matrix(target_test, pred2, class_names))

    except Exception as e:
        st.write(f"An error occurred during comparison: {e}")

def render_all_models_view(prepared_data):
    """
    Render comparison for all three models.
    """
    data_train, data_test, target_train, target_test, le = prepared_data
    class_names = ['high', 'low', 'moderate', 'very high', 'very low']
    
    # Name mappings for SHAP
    class_names_s = {0: 'high', 1: 'low', 2: 'moderate', 3: 'very high', 4: 'very low'}
    class_names_lgb = {0: 'very low', 1: 'very high', 2: 'high', 3: 'moderate', 4: 'low'}
    class_names_cat = {0: 'very low', 1: 'low', 2: 'high', 3: 'moderate', 4: 'very high'}

    try:
        bst = train_model(data_train, target_train, set_params_xgb())
        clf = train_model_lgb(data_train, target_train, set_params_lgb())
        cat_model = train_model_catboost(data_train, target_train, set_params_catboost())

        acc_xgb, pred_xgb = calculate_accuracy_and_predictions_general(bst, data_test, target_test, 'bst')
        acc_lgb, pred_lgb = calculate_accuracy_and_predictions_general(clf, data_test, target_test, 'clf')
        acc_cat, pred_cat = calculate_accuracy_and_predictions_general(cat_model, data_test, target_test, 'cat_model')

        plot_acc_bar_chart_full_comp(acc_xgb, acc_lgb, acc_cat, 'XGBoost', 'LightGBM', 'Catboost')

        if st.sidebar.checkbox('Create Sankey Diagram'):
            st.header('***Sankey Diagram Comparison:***')
            sankey_fig = generate_Sankey_three(pred_cat, pred_xgb, pred_lgb, target_test)
            st.plotly_chart(sankey_fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        if st.sidebar.checkbox('Calculate Shapley Values'):
            with col1:
                st.subheader('***Shapley Values for XGBoost:***')
                sv_x, _ = generate_Shapley(data_train, bst)
                shap.summary_plot(sv_x, data_train, plot_type="bar", class_names=list(class_names_s.values()), show=False)
                st.pyplot(bbox_inches='tight', pad_inches=0)
            with col2:
                st.subheader('***Shapley Values for LightGBM:***')
                sv_l, _ = generate_Shapley(data_train, clf)
                shap.summary_plot(sv_l, data_train, plot_type="bar", class_names=list(class_names_lgb.values()), show=False)
                st.pyplot(bbox_inches='tight', pad_inches=0)
            with col3:
                st.subheader('***Shapley Values for Catboost:***')
                sv_c, _ = generate_Shapley(data_train, cat_model)
                shap.summary_plot(sv_c, data_train, plot_type="bar", class_names=list(class_names_cat.values()), show=False)
                st.pyplot(bbox_inches='tight', pad_inches=0)

        if st.sidebar.checkbox('Generate Confusion Matrix'):
            with col1:
                st.subheader('***Confusion Matrix for XGBoost:***')
                st.pyplot(plot_confusion_matrix(target_test, pred_xgb, class_names))
            with col2:
                st.subheader('***Confusion Matrix for LightGBM:***')
                st.pyplot(plot_confusion_matrix(target_test, pred_lgb, class_names))
            with col3:
                st.subheader('***Confusion Matrix for Catboost:***')
                st.pyplot(plot_confusion_matrix(target_test, pred_cat, class_names))

    except Exception as e:
        st.write(f"An error occurred during all-model comparison: {e}")
