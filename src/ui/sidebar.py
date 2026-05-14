import streamlit as st
from src.data_loader import prepare_data

def render_sidebar(df, df_target):
    """
    Render the sidebar for dataset selection and model options.
    Returns:
        df_filtered (pd.DataFrame): The filtered dataframe.
        model_option (str): The selected model analysis option.
        diagram_option (str): The selected diagram comparison option.
        data_ready (bool): Whether the models have been trained and data is ready.
        prepared_data_tuple (tuple): (data_train, data_test, target_train, target_test, le) or None.
    """
    st.sidebar.write('Adjust Dataset:')
    
    # Column selection
    all_columns = df.columns.tolist()
    if st.sidebar.checkbox('Select All'):
        selected_columns = all_columns[0:]
    else:
        selected_columns = st.sidebar.multiselect('Select columns', all_columns, all_columns[0])
    
    df_filtered = df[selected_columns]
    
    # Training trigger
    data_ready = False
    prepared_data_tuple = None
    if st.sidebar.checkbox('Train Models'):
        data_train, data_test, target_train, target_test, le = prepare_data(df_filtered, df_target)
        prepared_data_tuple = (data_train, data_test, target_train, target_test, le)
        st.sidebar.write('Dataset selected')
        data_ready = True

    # Model selection
    model_option = st.sidebar.selectbox(
        'Choose a model for analysis:', 
        ['Choose Model', 'XGBoost', 'LightGBM', 'CatBoost', 'XGBoost vs LightGBM', 'XGBoost vs Catboost', 'LightGBM vs Catboost', 'All Models']
    )
    
    # Diagram comparison selection
    diagram_option = st.sidebar.selectbox(
        'Choose a diagram for comparison:', 
        ['Choose Diagram', 'Sankey', 'Shapley', 'Matrix']
    )

    return df_filtered, model_option, diagram_option, data_ready, prepared_data_tuple
