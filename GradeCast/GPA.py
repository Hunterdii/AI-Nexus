import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset = pd.read_csv("GradeCast/Student_performance_data _.csv")  # Corrected filename

# Feature Engineering
dataset['StudyGradeInteraction'] = dataset['StudyTimeWeekly'] * dataset['GradeClass']

# Prepare features and target variable
x = dataset[['StudentID', 'Age', 'StudyTimeWeekly', 'GradeClass', 'StudyGradeInteraction']]
y = dataset['GPA']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Hyperparameter Tuning with KNN
param_grid = {'n_neighbors': np.arange(1, 20)}
knn_regressor = KNeighborsRegressor()
grid_search = GridSearchCV(knn_regressor, param_grid, cv=5)
grid_search.fit(x_train_scaled, y_train)

best_knn_regressor = grid_search.best_estimator_
best_knn_regressor.fit(x_train_scaled, y_train)

# Save the model
joblib.dump(best_knn_regressor, "GradeCast/student_gpa_model.pkl")

# Streamlit UI
st.set_page_config(page_title="Student GPA Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Student GPA Prediction App")

st.markdown("""
    Welcome to the enhanced Student GPA Prediction app! This application uses a K-Nearest Neighbors regression model with hyperparameter tuning to predict the GPA of a student based on various characteristics. 
    Explore different visualizations and features to get insights into the model's performance.
""")

# Sidebar for additional options
with st.sidebar:
    st.header("🔧 Settings")
    st.markdown("Adjust the app settings below:")
    
    n_neighbors = st.slider("Number of Neighbors (K in KNN)", 1, 20, best_knn_regressor.n_neighbors)
    best_knn_regressor.n_neighbors = n_neighbors

    show_dataset = st.checkbox("Show Dataset Overview", value=True)
    show_corr_heatmap = st.checkbox("Show Feature Correlation Heatmap", value=True)
    show_scatterplot = st.checkbox("Show GPA vs Study Time Weekly", value=True)
    show_performance = st.checkbox("Show Model Performance", value=True)

# Input form
with st.form("prediction_form"):
    st.subheader("🔍 Enter Student Details")
    StudentID = st.number_input("Student ID", min_value=1, step=1, value=3392)
    Age = st.number_input("Age", min_value=1, step=1, value=16)
    StudyTimeWeekly = st.number_input("Study Time Weekly (hours)", min_value=0.0, value=17.8, step=0.5)
    GradeClass = st.number_input("Grade Class", min_value=1.0, value=1.0, step=0.5)
    submit_button = st.form_submit_button(label='🎓 Predict GPA')

# Prediction and output
if submit_button:
    model = joblib.load("GradeCast/student_gpa_model.pkl")  # Ensure this path is correct
    x_input = np.array([StudentID, Age, StudyTimeWeekly, GradeClass, StudyTimeWeekly * GradeClass])
    
    if any(x_input <= 0):
        st.warning("⚠️ Input values must be greater than 0")
    else:
        x_scaled = scaler.transform([x_input])
        predicted_gpa = model.predict(x_scaled)
        st.success(f"🎉 Predicted GPA: **{predicted_gpa[0]:.6f}**")

# Data visualization
if show_dataset:
    st.markdown("### 📊 Dataset Overview & Visualization")
    st.write(dataset.head())
    
    # Histograms for feature distributions
    st.markdown("#### 🔎 Feature Distributions")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    sns.histplot(dataset['Age'], bins=20, ax=axes[0, 0], kde=True).set(title='Age Distribution')
    sns.histplot(dataset['StudyTimeWeekly'], bins=20, ax=axes[0, 1], kde=True).set(title='Study Time Weekly Distribution')
    sns.histplot(dataset['GradeClass'], bins=10, ax=axes[0, 2], kde=True).set(title='Grade Class Distribution')
    sns.histplot(dataset['StudyGradeInteraction'], bins=20, ax=axes[1, 0], kde=True).set(title='Study-Grade Interaction Distribution')
    sns.histplot(dataset['GPA'], bins=20, ax=axes[1, 1], kde=True).set(title='GPA Distribution')
    plt.tight_layout()
    st.pyplot(fig)

if show_scatterplot:
    st.markdown("#### GPA vs Study Time Weekly")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=dataset, x='StudyTimeWeekly', y='GPA', hue='GradeClass', palette='viridis')
    st.pyplot(plt)

if show_corr_heatmap:
    st.markdown("### 🔍 Feature Correlation Heatmap")
    plt.figure(figsize=(12, 8))
    correlation_matrix = dataset[['Age', 'StudyTimeWeekly', 'GradeClass', 'StudyGradeInteraction', 'GPA']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(plt)

# Model Performance
if show_performance:
    st.markdown("### 📈 Model Performance")
    
    # Cross-validation score
    cross_val_scores = cross_val_score(best_knn_regressor, x_train_scaled, y_train, cv=5)
    st.markdown(f"**Mean Cross-Validation R² Score:** {np.mean(cross_val_scores):.2f}")
    
    # Test set performance
    y_pred_train = best_knn_regressor.predict(x_train_scaled)
    y_pred_test = best_knn_regressor.predict(x_test_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    st.markdown(f"**Train RMSE:** {train_rmse:.2f}")
    st.markdown(f"**Test RMSE:** {test_rmse:.2f}")
    st.markdown(f"**Train R² Score:** {train_r2:.2f}")
    st.markdown(f"**Test R² Score:** {test_r2:.2f}")
    
    # Accuracy Bar
    accuracy = test_r2 * 100
    st.markdown("### 🏆 Model Accuracy")
    st.progress(int(accuracy))

    # Confusion Matrix for Binned GPA Prediction
    st.markdown("#### 📊 Confusion Matrix for Binned GPA Predictions")
    
    # Bin the GPA values
    bins = [0, 2, 2.5, 3, 3.5, 4]  # Adjust the bins according to the data distribution
    y_test_binned = np.digitize(y_test, bins) - 1
    y_pred_binned = np.digitize(y_pred_test, bins) - 1

    # Calculate confusion matrix
    cm = confusion_matrix(y_test_binned, y_pred_binned, labels=range(len(bins) - 1))

    # Ensure that display_labels has one less item than bins, since bins define the edges
    display_labels = [f'{bins[i]}-{bins[i + 1]}' for i in range(len(bins) - 1)]

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels).plot(ax=ax)
    st.pyplot(fig)
