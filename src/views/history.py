import streamlit as st
from src.utils import compareCharts

def render_history_view(diagram_option):
    """
    Render the historical diagram comparison section at the bottom.
    """
    if diagram_option == 'Sankey':
        compareCharts('sankey_chart.png')
    elif diagram_option == 'Shapley':
        compareCharts('shapley_chart.png')
    elif diagram_option == 'Matrix':
        compareCharts('matrix_chart.png')
    else:
        pass
