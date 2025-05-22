Project Overview
This Python script is designed for analyzing the U.S. Accidents dataset, performing data preprocessing, feature selection, model training, and evaluation to predict accident severity. Your approach includes data cleaning, transformation, feature engineering, model selection, hyperparameter tuning, and dimensionality reduction. The project follows a structured machine learning pipeline.

How to Run the Code
- Open the Colab Link
- If the script contains a Colab link, simply click on it or copy-paste the URL into your browser.
- This will open the notebook in Google Colab, an interactive cloud-based environment.

Project Objectives
The primary aim of this project is to:
- Analyze road accident data: Understand patterns in accident severity distribution.
- Preprocess the dataset: Handle missing values and categorical encoding.
- Select important features: Identify key attributes influencing & road accidents and its severity.
- Train machine learning models: Compare multiple classifiers.
- Optimize models: Tune hyperparameters for better performance.
- Evaluate results: Visualize confusion matrices, ROC curves, and feature importance.
- Apply PCA: Reduce dimensionality for efficiency.

Data Preprocessing Steps
- Loading and Inspecting Data:
- Reads the dataset using pandas (pd.read_csv).
- Converts Start_Time column to datetime format.
- Displays rows with parsing errors (.isna() check).
- Filtering Data:
- Selects data from 2016 to 2023 (10,000 rows per year).
- Concatenates filtered data for final analysis.
- Handling Missing Values:
- Calculates missing percentage for each column.
- Removes columns with missing values >10%.
- Feature Engineering:
- Removes non-informative columns (e.g., ID, Zipcode, Airport_Code).
- Filters city/county/state entries with occurrences <100.
- Extracts time-based features (Start_Hour, Start_Year, End_Hour, End_Year).
- Converts categorical features into numerical encoding using LabelEncoder.
- Feature Imputation:
- Uses SimpleImputer to fill missing values based on datatype (median for numerical, most_frequent for categorical).

Machine Learning Models Used
Your project uses multiple classifiers to predict accident severity:
- XGBoost (XGBClassifier):
- Powerful ensemble method using gradient boosting.
- Robust to missing data and handles large-scale datasets.
- Random Forest (RandomForestClassifier):
- Ensemble of decision trees improving accuracy.
- Less prone to overfitting compared to individual trees.
- Gradient Boosting (GradientBoostingClassifier):
- Sequentially improves weak learners.
- Often performs better in complex data patterns.
- Multi-Layer Perceptron (MLPClassifier):
- Neural network-based classification model.
- Trained with backpropagation for deep learning capabilities.

Model Training & Evaluation
- Splitting the Data:
- Uses train_test_split (80% training, 20% testing).
- Applies StandardScaler for normalization.
- Model Training:
- Each classifier is trained on the normalized dataset.
- cross_val_score is used to validate performance across folds.
- Performance Metrics:
- Accuracy, Precision, Recall, F1-score (for classification report).
- ROC Curve and AUC Score (to measure classification ability).
- Confusion Matrix (visualizes true positives and false predictions).

Feature Selection & Dimensionality Reduction
- Mutual Information (SelectKBest):
- Identifies top 10 most important features.
- Plots feature importance.
- Principal Component Analysis (PCA):
- Reduces dimensionality to 10 principal components.
- Improves model efficiency with lower computation costs.

Hyperparameter Tuning
- GridSearchCV is applied for fine-tuning:
- XGBoost: Adjusts learning_rate, max_depth, n_estimators.
- Random Forest: Adjusts min_samples_split, max_depth, n_estimators.
- Gradient Boosting: Adjusts learning_rate, max_depth, n_estimators.
- MLP: Adjusts activation function, learning rate, hidden layers.
