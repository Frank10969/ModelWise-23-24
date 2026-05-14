<div align="center">

# ModelWise

An advanced **Streamlit dashboard** for soil organic carbon prediction and model interpretability.

<br />

![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-000000?style=flat)
![LightGBM](https://img.shields.io/badge/LightGBM-FFB900?style=flat)
![CatBoost](https://img.shields.io/badge/CatBoost-FF4620?style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

<hr>

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Project Background](#project-background) • [Milestones](#milestones)

<hr>

</div>

## Project Background

### What was the Problem?
*   **Understanding the Data:** Soil organic carbon datasets are complex and require deep exploration.
*   **Model Selection:** Finding the best classification model for various data subsets.
*   **Black-Box Models:** Predictions often lack transparency, making it hard to trust the results.

### The Solution
ModelWise addresses these issues by implementing **Explainable AI (XAI)**. We move beyond simple predictions to understand *why* models behave the way they do, using high-fidelity visualizations and interactive comparisons.

## Features
- **📊 Interactive Comparison:** Compare accuracy, confusion matrices, and SHAP values across **XGBoost**, **LightGBM**, and **CatBoost**.
- **🔍 Explainable AI (XAI):** 
    - **SHAP (Shapley Additive Explanations):** Deep dive into feature importance.
    - **Sankey Diagrams:** Visualize the flow from predicted to actual classes.
- **⚙️ Dynamic Dataset Selection:** Fine-tune the analysis by choosing specific data subsets and features.
- **📈 Advanced Visualizations:** Real-time generation of accuracy bar charts, matrices, and flow diagrams.

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
    The project expects the following files in the `ModelWise-23-24/data/` directory:
    - `data/lucas_organic_carbon_training_and_test_data.csv`
    - `data/lucas_organic_carbon_target.csv`
    *(Note: Large data files are gitignored and must be added manually.)*

## Usage
Run the modular dashboard with a single command:
```bash
streamlit run app.py
```

## Milestones
1.  **Exploration of Data:** Initial analysis and understanding of the SOC dataset.
2.  **Visualization Research:** Exploring effective XAI methods (Sankey, Confusion Matrices).
3.  **SHAP Integration:** Implementing feature-level importance calculations.
4.  **XAI Strategy:** Developing the conceptual framework for model transparency.
5.  **Modular Implementation:** Final refactoring into a clean, component-based architecture.

## Authors
- **Minh Hoang Do**
- **Khaled Ghazi**
- **Benedict da Silva Araújo Finsterwalder**
- **Frank Hachenberger**

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
