# AutoML and AutoDL Project

This project provides a user-friendly interface for utilizing AutoML and AutoDL capabilities, built with Streamlit. It allows users to easily train machine learning and deep learning models without extensive coding or expertise.

## Features

* **Automated Machine Learning (AutoML)**
    *  Automated model selection and hyperparameter tuning for classification and regression tasks.
    *  Support for various popular algorithms (e.g., Random Forest, XGBoost, Logistic Regression).
    *  Data preprocessing and feature engineering options.
* **Automated Deep Learning (AutoDL)**
    *  Automated neural architecture search for image classification and object detection.
    *  Transfer learning with pre-trained models.
    *  Data augmentation options for improved performance.
* **Streamlit UI**
    *  Intuitive interface for data upload, model training, and result visualization.
    *  Real-time progress updates and model performance metrics.
    *  Downloadable trained models for deployment.

## Technologies Used

* **AutoML Libraries:**
    *  PyCaret
    *  Auto-sklearn
    *  TPOT
* **AutoDL Libraries:**
    *  AutoKeras
    *  Google AutoML
* **Framework:** Streamlit
* **Languages:** Python

## Project Structure

```
├── app.py             # Main Streamlit application file
├── utils              # Utility functions for data preprocessing, etc.
├── models             # Directory to store trained models
├── data               # Sample datasets (optional)
└── requirements.txt   # Project dependencies
```

## Installation

1. Clone the repository: `git clone https://github.com/utkarsh-cpu/automl-autodl-streamlit.git`
2. Install dependencies: `pip install -r requirements.txt`

## Usage

1. Run the Streamlit app: `streamlit run app.py`
2. Upload your dataset.
3. Select the task type (classification, regression, image classification, etc.).
4. Choose AutoML or AutoDL.
5. Configure any optional settings.
6. Start the training process.
7. View results and download the trained model.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

[Specify your license, e.g., MIT License]

## Acknowledgements

* [List any resources, libraries, or individuals you want to acknowledge]
