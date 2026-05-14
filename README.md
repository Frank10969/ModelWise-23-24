# ModelWise: Interactive Comparison of Classification Models

ModelWise is a professional Streamlit-based dashboard designed to compare and analyze various machine learning models (XGBoost, LightGBM, CatBoost) for predicting soil organic carbon. The project leverages Explainable AI (XAI) to provide transparent, non-"black-box" insights into model behavior and predictions.

## Project Background

### What was the Problem?
*   **Understanding the data:** Soil organic carbon datasets can be complex and multidimensional.
*   **Training effective models:** Finding the right classification model for specific data subsets.
*   **Model Interpretability:** Moving beyond black-box predictions to understand *why* a model makes a certain decision (XAI).

### Why address this?
As Software Engineers and Data Scientists, it is crucial to understand the behavior of our models. We aim to visualize results clearly and ensure that predictions are grounded in interpretable logic.

## Features
- **Interactive Model Comparison:** Compare accuracy, confusion matrices, and SHAP values across multiple models (XGBoost, LightGBM, CatBoost).
- **Explainable AI (XAI):** 
    - **SHAP (Shapley Additive Explanations):** Visualizing feature importance.
    - **Sankey Diagrams:** Illustrating the flow between actual and predicted classes.
- **Dynamic Dataset Selection:** Users can choose individual features, leading to a better understanding of how specific variables impact model performance.
- **Advanced Visualizations:**
    - Accuracy Bar Charts
    - Confusion Matrices
    - Interactive Sankey Diagrams

## XAI Strategy & Implementation
The application follows a user-centric XAI strategy:
1.  **Input:** User chooses features and models to compare.
2.  **Interaction:** User can change the dataset and see real-time changes in model behavior.
3.  **Visualizations:** Includes 4 different visualizations (Accuracy, Sankey, SHAP, Confusion Matrix).
4.  **Comparisons:** 
    *   Compare 2 or 3 models on the same dataset.
    *   Analyze a single model on two different datasets.

## Milestones
1.  **Exploration of Data:** In-depth understanding of the soil organic carbon dataset.
2.  **Visualization Exploration:** Researching effective methods like Sankey and Confusion Matrices.
3.  **SHAP Integration:** Adding Shapley-Values for feature-level interpretability.
4.  **XAI Concept Development:** Formalizing the presentation of explainability.
5.  **High-Fidelity Design:** Implementing the final Streamlit application and visualizations.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ModelWise-23-24.git
    cd ModelWise-23-24
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Requirements:**
    The project expects CSV files in the `ModelWise-23-24/data/` directory:
    - `data/lucas_organic_carbon_training_and_test_data.csv`
    - `data/lucas_organic_carbon_target.csv`
    *(Note: The data directory is mostly excluded from the repository via .gitignore)*

## Usage
Run the dashboard using Streamlit:
```bash
streamlit run app.py
```

## Conclusion & Insights

### Pros
- Comprehensive comparisons between models and datasets.
- Flexible feature selection.
- Simple and intuitive UI design.
- Open-Source and modular architecture.

### Cons
- No direct exploration of datasets (e.g., boxplots, scatterplots).
- Current lack of model caching/downloading.
- Opportunities for further runtime optimization.

### Lessons Learned
- Mastery of Streamlit for rapid dashboard development.
- Deep dive into Sankey and SHAP for XAI.
- Importance of high-quality data and exploration methods.
- The challenge of balancing feature additions with stability and user-friendliness.

## Authors
- Minh Hoang Do
- Khaled Ghazi
- Benedict da Silva Araújo Finsterwalder
- Frank Hachenberger

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
