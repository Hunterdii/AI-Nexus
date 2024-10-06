import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load the dataset
dataset = pd.read_csv("Iriswise/Iris.csv")

# Prepare the features and target variable
x = dataset.drop(["Species", "Id"], axis=1)
y = dataset["Species"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Initialize and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

# Save the model if not saved already
model_file = "Iriswise/classifier.pkl"
try:
    knn_model = joblib.load(model_file)
except (FileNotFoundError, AttributeError):
    joblib.dump(knn, model_file)
    knn_model = knn  # Save it again and load it

# Streamlit UI
st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="wide")

# Custom Styling and Animation
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        font-family: 'Roboto', sans-serif;
    }
    .header {
        background-color: #263238;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: #ffffff;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        animation: fadeIn 2s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .sidebar .sidebar-content {
        background-color: #37474f;
        color: #ffffff;
    }
    .expander {
        background-color: #455a64;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #004d40;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #00796b;
    }
    .welcome-text {
        background-color: #455a64;
        padding: 20px;
        border-radius: 10px;
        color: #ffffff;
        text-align: center;
        font-size: 1.2em;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .tooltip {
        position: relative;
         display: inline-block;
         cursor: pointer;
    }
    .tooltip .tooltiptext {
         visibility: hidden;
        width: 200px;
         background-color: #333;
         color: #fff;
         text-align: center;
         border-radius: 5px;
         padding: 5px;
         position: absolute;
         z-index: 1;
        bottom: 125%;
        left: 50%;
         margin-left: -100px;
         opacity: 0;
         transition: opacity 0.3s;
     }
     .tooltip:hover .tooltiptext {
         visibility: visible;
         opacity: 1;
     }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h2>🌼 Welcome to the Iris Species Prediction App 🌺</h2></div>', unsafe_allow_html=True)

# Sidebar for additional options
with st.sidebar:
    st.header("🔧 Settings")
    st.markdown("Adjust the app settings below:")
    n_neighbors = st.slider("Neighbors", 1, 15, 3)
    knn_model.n_neighbors = n_neighbors
    knn_model.fit(x_train, y_train)  # Re-train with new K

    # Checkboxes for additional features
    show_dataset = st.checkbox("Show Dataset Overview 🗂️", value=True)
    show_pairplot = st.checkbox("Show Pairplot Visualization 📊", value=True)
    show_performance = st.checkbox("Show Model Performance 📈", value=True)
    show_confusion_matrix = st.checkbox("Show Confusion Matrix 🧩", value=True)
    show_model_summary = st.checkbox("Show Model Summary 📝", value=True)

# Input form
with st.form("prediction_form"):
    st.subheader("🔍 Enter Flower Features")
    Sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.0, step=0.1)
    Sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.0, step=0.1)
    Petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.5, step=0.1)
    Petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2, step=0.1)
    submit_button = st.form_submit_button(label='🌟 Predict Species')

# Prediction and output
if submit_button:
    with st.spinner("Predicting..."):
        x_input = pd.DataFrame([[Sepal_length, Sepal_width, Petal_length, Petal_width]], 
                               columns=x.columns)
        if any(x_input.values[0] <= 0):
            st.warning("⚠️ Input values must be greater than 0")
        else:
            prediction = knn_model.predict(x_input)
            species_images = {
                'Iris-setosa': 'Iriswise/assets/Irissetosa1.jpg',
                'Iris-versicolor': 'Iriswise/assets/Versicolor.webp',
                'Iris-virginica': 'Iriswise/assets/virgina.jpg'
            }
            st.success(f"🎉 Predicted Species: **{prediction[0]}**")
            st.image(species_images[prediction[0]], caption=f'Iris {prediction[0]}', use_column_width=True)

# Data visualization with expanders
if show_dataset:
    with st.expander("🌸 Dataset Overview & Visualization", expanded=False):
        st.write(dataset.head())
        st.markdown("#### Data Distribution")
        st.bar_chart(dataset["Species"].value_counts())

if show_pairplot:
    with st.expander("📊 Pairplot of Iris Features", expanded=False):
        fig = px.scatter_matrix(dataset, dimensions=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
                                color="Species", symbol="Species", title="Pairplot of Iris Features",
                                labels={"SepalLengthCm": "Sepal Length (cm)", "SepalWidthCm": "Sepal Width (cm)",
                                        "PetalLengthCm": "Petal Length (cm)", "PetalWidthCm": "Petal Width (cm)"},
                                hover_name="Species")
        st.plotly_chart(fig)

# Model performance metrics
if show_performance:
    with st.expander("📈 Model Performance", expanded=False):
        train_accuracy = knn_model.score(x_train, y_train) * 100
        test_accuracy = knn_model.score(x_test, y_test) * 100
        
        st.markdown(f"**Train Set Accuracy:** {train_accuracy:.2f}%")
        st.markdown(f"**Test Set Accuracy:** {test_accuracy:.2f}%")

        if show_confusion_matrix:
            st.markdown("### 🧩 Confusion Matrix")
            cm = confusion_matrix(y_test, knn_model.predict(x_test), labels=knn_model.classes_)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=knn_model.classes_, yticklabels=knn_model.classes_, ax=ax)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            st.pyplot(fig)

# Model Summary
if show_model_summary:
    with st.expander("📋 Model Summary", expanded=False):
        st.write(f"**Number of Neighbors:** {knn.n_neighbors}")
        st.write(f"**Algorithm:** {knn._fit_method}")
        
        distance_metric = knn.metric
        if distance_metric == 'minkowski' and knn.p == 2:
            distance_metric = 'Euclidean (Minkowski with p=2)'
        elif distance_metric == 'minkowski':
            distance_metric = f'Minkowski with p={knn.p}'
            
        st.write(f"**Distance Metric:** {distance_metric}")
        st.write("This summary provides an overview of the current model configuration.")


