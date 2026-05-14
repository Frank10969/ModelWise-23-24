import os
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

def log_message(message):
    """
    Log a message to the console.
    """
    print(message)

def get_file_modification_timestamp(file_path):
    """
    Get the modification timestamp of a file.
    """
    if os.path.exists(file_path):
        return os.path.getmtime(file_path)
    else:
        log_message("File not found")
        return 0.0

def save_image_to_folder(fig, folder_path, image_name):
    """
    Save a Plotly figure or Matplotlib figure to a specified folder.
    """
    image_file_path = os.path.join(folder_path, image_name)
    if isinstance(fig, go.Figure):
        pio.write_image(fig, image_file_path)
    else:
        # Handle Matplotlib figure (e.g., from SHAP)
        fig.savefig(image_file_path, bbox_inches='tight')

def checkFoldersAndSaveNewImage(fig, file_name):
    """
    Check multiple folders and save the image in the most appropriate one.
    """
    image_file_path1 = os.path.join('folder1', file_name)
    timestamp1 = get_file_modification_timestamp(image_file_path1)
    log_message("Timestamp 1: " + str(timestamp1))
    
    image_file_path2 = os.path.join('folder2', file_name)
    timestamp2 = get_file_modification_timestamp(image_file_path2)
    log_message("Timestamp 2: " + str(timestamp2))

    if timestamp1 == 0.0:
        save_image_to_folder(fig, 'folder1', file_name)
        log_message("Saved in folder1 (empty)")
        return

    if timestamp2 == 0.0:
        save_image_to_folder(fig, 'folder2', file_name)
        log_message("Saved in folder2 (empty)")
        return

    if (timestamp1 - timestamp2 < 0):
        save_image_to_folder(fig, 'folder1', file_name)
        log_message("Saved in folder1 (older)")
    elif (timestamp1 - timestamp2 > 0):
        save_image_to_folder(fig, 'folder2', file_name)
        log_message("Saved in folder2 (older)")
    else:
        save_image_to_folder(fig, 'folder1', file_name)
        log_message("Saved in folder1 (default)")

def createSaveButton(fig, file_name):
    """
    Create a Streamlit button to save a diagram.
    """
    if st.sidebar.button(f'Save {file_name}'):
        checkFoldersAndSaveNewImage(fig, file_name)
        log_message("Save operation finished")
        st.write('Diagram saved!')

def compareCharts(file_name):
    """
    Compare two versions of a chart saved in different folders.
    """
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
        st.header('***Before:***')
        st.image(final_image_file_path1, use_container_width=True)
    with col2:
        st.header('***After:***')
        st.image(final_image_file_path2, use_container_width=True)
