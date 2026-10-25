Cardiovascular Disease Prediction

📌 Project Overview

This project uses Machine Learning algorithms to analyze cardiovascular disease data and predict the presence of heart disease. It includes data preprocessing, exploratory data analysis, data visualization, and model performance comparison.

🎯 Objectives

- Clean and preprocess the dataset.
- Analyze cardiovascular health data.
- Visualize important features using graphs.
- Generate a correlation matrix.
- Compare five Machine Learning algorithms.
- Evaluate models using accuracy and a confusion matrix.

🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

🤖 Machine Learning Algorithms

1. Support Vector Machine (SVM)
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Logistic Regression
5. Random Forest

📂 Project Structure

Cardiovascular-Disease-Prediction/
├── main.py
├── cardio_train.csv
├── README.md
├── model_accuracy.csv
├── best_heart_disease_model.pkl
└── plots/
    ├── target_distribution.png
    ├── age_distribution.png
    ├── gender_disease.png
    ├── cholesterol_disease.png
    ├── blood_pressure.png
    ├── correlation_matrix.png
    ├── model_accuracy.png
    └── confusion_matrix.png

⚙️ Installation and Setup

1. Install Required Libraries

python -m pip install pandas numpy matplotlib seaborn scikit-learn joblib

2. Prepare the Dataset

Place "cardio_train.csv" in the project folder.

3. Run the Project

python main.py

📊 Project Outputs

- Dataset analysis and cleaning
- Data visualization graphs
- Feature correlation heatmap
- Model accuracy comparison
- Classification report
- Confusion matrix
- Saved best-performing model

📈 Model Evaluation

The project compares the performance of five Machine Learning algorithms. Actual accuracy values are generated when the program runs and are saved in "model_accuracy.csv".

⚠️ Disclaimer

This project is developed for educational purposes only. It is not intended to replace professional medical advice or diagnosis.
