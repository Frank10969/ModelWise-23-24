import streamlit as st
import warnings
import os

# Import modularized logic
from src.data_loader import load_data
from src.ui.sidebar import render_sidebar
from src.views.single_model import render_single_model_view
from src.views.comparison import render_comparison_view, render_all_models_view
from src.views.history import render_history_view
from src.utils import log_message

def main():
    # Page configuration
    st.set_page_config(layout="wide")
    st.title('ModelWise 23/24')

    # 1. Load raw data
    try:
        df, df_target = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.write("Ensure your CSV files are in the 'data/' folder.")
        return

    # 2. Render Sidebar and handle state
    df_filtered, model_option, diagram_option, data_ready, prepared_data = render_sidebar(df, df_target)

    # 3. Create temporary folders for diagram comparisons
    for folder_name in ['folder1', 'folder2']:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            log_message(f"Folder {folder_name} created.")

    warnings.filterwarnings('ignore')

    # 4. Render Main Content based on selection
    if data_ready and prepared_data:
        if model_option == 'Choose Model':
            st.info("Please choose a model analysis option from the sidebar.")
        elif 'vs' in model_option:
            render_comparison_view(prepared_data, model_option)
        elif model_option == 'All Models':
            render_all_models_view(prepared_data)
        else:
            render_single_model_view(prepared_data, model_option)
    else:
        st.info("Please train the models by clicking 'Train Models' in the sidebar.")

    # 5. Render Diagram Comparison History (Bottom section)
    if diagram_option != 'Choose Diagram':
        st.divider()
        st.subheader(f"Comparison History: {diagram_option}")
        render_history_view(diagram_option)

if __name__ == "__main__":
    main()
