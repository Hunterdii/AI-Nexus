# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.impute import SimpleImputer
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, f1_score
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# import pickle

# dataset = pd.read_csv("diabetes.csv")

# columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
# imputer = SimpleImputer(missing_values=0, strategy='median')
# dataset[columns_to_impute] = imputer.fit_transform(dataset[columns_to_impute])

# X = dataset.drop(["Outcome"], axis=1)
# y = dataset["Outcome"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# poly = PolynomialFeatures(degree=2, interaction_only=True)
# X_train_poly = poly.fit_transform(X_train)
# X_test_poly = poly.transform(X_test)

# st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide")
# st.title("🩺 Diabetes Prediction App")

# st.markdown("""
#     Welcome to the advanced version of the Diabetes Prediction App! This version allows you to choose from multiple algorithms, 
#     tune hyperparameters, and get detailed model performance insights.
# """)

# with st.sidebar:
#     st.header("🔧 Model Settings")
#     model_choice = st.selectbox("Select Model", ["K-Nearest Neighbors", "Logistic Regression", "Linear Regression", 
#                                                  "Random Forest", "Gradient Boosting", "SVM", "XGBoost"])
    
#     if model_choice == "K-Nearest Neighbors":
#         n_neighbors = st.slider("Number of Neighbors (K in KNN)", 1, 15, 3)
#         model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
#     elif model_choice == "Logistic Regression":
#         c_value = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
#         model = LogisticRegression(C=c_value, random_state=42)
    
#     elif model_choice == "Linear Regression":
#         model = LinearRegression()
    
#     elif model_choice == "Random Forest":
#         n_estimators = st.slider("Number of Trees", 10, 200, 100)
#         max_depth = st.slider("Max Depth of Trees", 2, 20, 10)
#         model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
#     elif model_choice == "Gradient Boosting":
#         learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
#         n_estimators = st.slider("Number of Estimators", 50, 200, 100)
#         model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=42)
    
#     elif model_choice == "SVM":
#         c_value = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
#         kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
#         model = SVC(C=c_value, kernel=kernel, probability=True, random_state=42)
    
#     elif model_choice == "XGBoost":
#         learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
#         n_estimators = st.slider("Number of Estimators", 50, 200, 100)
#         max_depth = st.slider("Max Depth of Trees", 3, 10, 5)
#         model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
#     show_dataset = st.checkbox("Show Dataset Overview", value=True)
#     show_pairplot = st.checkbox("Show Pairplot Visualization", value=False)
#     show_performance = st.checkbox("Show Model Performance", value=True)
#     show_advanced_metrics = st.checkbox("Show Advanced Metrics (F1-Score, AUC-ROC)", value=True)

# model.fit(X_train_poly, y_train)

# # Save the model
# pickle_out = open("classifier.pkl", "wb")
# pickle.dump(model, pickle_out)
# pickle_out.close()

# with st.form("prediction_form"):
#     st.subheader("🔍 Enter Patient's Health Metrics")
#     Pregnancies = st.number_input("Pregnancies", min_value=0, value=1, step=1)
#     Glucose = st.number_input("Glucose Level", min_value=0.0, value=120.0, step=1.0)
#     BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, value=70.0, step=1.0)
#     SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0.0, value=20.0, step=1.0)
#     Insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0.0, value=80.0, step=1.0)
#     BMI = st.number_input("BMI", min_value=0.0, value=30.0, step=0.1)
#     DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5, step=0.01)
#     Age = st.number_input("Age", min_value=1, value=30, step=1)
#     submit_button = st.form_submit_button(label='🌟 Predict Diabetes')

# if submit_button:
#     input_data = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
#     input_data = input_data.reshape(1, -1)
#     input_data = poly.transform(scaler.transform(input_data))
#     prediction = model.predict(input_data)
    
#     if model_choice == "Linear Regression":
#         prediction = [1 if p > 0.5 else 0 for p in prediction]  # Convert probabilities to binary outcome
    
#     result = "Positive for Diabetes" if prediction[0] == 1 else "Negative for Diabetes"
#     st.success(f"🎉 Prediction: **{result}**")

# if show_dataset:
#     st.markdown("### 📊 Dataset Overview & Visualization")
#     st.write(dataset.head())

# if show_pairplot:
#     st.markdown("#### 🌸 Pairplot of Diabetes Features")
#     sns.pairplot(dataset, hue="Outcome", palette="coolwarm")
#     st.pyplot(plt)

# if show_performance:
#     st.markdown("### 📈 Model Performance")

#     y_pred = model.predict(X_test_poly)
#     if model_choice == "Linear Regression":
#         y_pred = [1 if p > 0.5 else 0 for p in y_pred]  # Convert probabilities to binary outcome
    
#     accuracy = accuracy_score(y_test, y_pred)
#     st.markdown(f"**Test Set Accuracy:** {accuracy:.2f}")
    
#     st.markdown("#### Classification Report")
#     st.text(classification_report(y_test, y_pred))

#     st.markdown("#### Confusion Matrix")
#     cm = confusion_matrix(y_test, y_pred)
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     st.pyplot(plt)

#     if show_advanced_metrics:
#         st.markdown("### ⚖️ Advanced Metrics")
#         f1 = f1_score(y_test, y_pred)
#         st.markdown(f"**F1-Score:** {f1:.2f}")

#         if model_choice != "Linear Regression":
#             st.markdown("#### ROC-AUC Curve")
#             y_pred_prob = model.predict_proba(X_test_poly)[:, 1]
#             fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
#             roc_auc = auc(fpr, tpr)
#             st.markdown(f"**ROC-AUC Score:** {roc_auc:.2f}")

#             plt.figure(figsize=(8, 6))
#             plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
#             plt.plot([0, 1], [0, 1], color='red', linestyle='--')
#             plt.xlim([0.0, 1.0])
#             plt.ylim([0.0, 1.05])
#             plt.xlabel('False Positive Rate')
#             plt.ylabel('True Positive Rate')
#             plt.title('Receiver Operating Characteristic (ROC)')
#             plt.legend(loc="lower right")
#             st.pyplot(plt)



# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.impute import SimpleImputer
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve, auc
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import plotly.graph_objects as go
# import shap
# import pickle
# import requests
# import streamlit_lottie as st_lottie

# # Load the dataset
# dataset = pd.read_csv("diabetes.csv")

# # Impute missing values
# columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
# imputer = SimpleImputer(missing_values=0, strategy='median')
# dataset[columns_to_impute] = imputer.fit_transform(dataset[columns_to_impute])

# # Split the dataset
# X = dataset.drop(["Outcome"], axis=1)
# y = dataset["Outcome"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Standardize the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Apply Polynomial Features
# poly = PolynomialFeatures(degree=2, interaction_only=True)
# X_train_poly = poly.fit_transform(X_train)
# X_test_poly = poly.transform(X_test)

# # Set Streamlit page configuration
# st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide")

# # Custom CSS for styling
# st.markdown(
#     """
#     <style>
#     body {
#         background-color: #f0f2f6;
#     }
#     .stApp {
#         background-color: #f0f2f6;
#         color: #333333;
#     }
#     .css-1aumxhk {
#         background-color: #3E65A5;
#         color: white;
#     }
#     .css-qri22k {
#         background-color: #3E65A5;
#         color: white;
#     }
#     div.stButton > button:first-child {
#         background-color: #3E65A5;
#         color:#FFFFFF;
#         height:3em;
#         width:100%;
#         border-radius:10px;
#         border:2px solid #f0f2f6;
#         font-size:20px;
#         font-weight:bold;
#         margin-top: 20px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # Load Lottie animation
# def load_lottie_url(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

# lottie_diabetes = load_lottie_url("https://assets8.lottiefiles.com/packages/lf20_jcikwtux.json")

# st.title("🩺 Diabetes Prediction App")
# st.markdown("""
#     Welcome to the advanced version of the Diabetes Prediction App! This version allows you to choose from multiple algorithms, 
#     tune hyperparameters, and get detailed model performance insights.
# """)

# # Sidebar for model selection and settings
# with st.sidebar:
#     st.header("🔧 Model Settings")
#     st_lottie.st_lottie(lottie_diabetes, height=200, width=200)

#     model_choice = st.selectbox("Select Model", ["K-Nearest Neighbors", "Logistic Regression", "Linear Regression", 
#                                                  "Random Forest", "Gradient Boosting", "SVM", "XGBoost"])
    
#     if model_choice == "K-Nearest Neighbors":
#         n_neighbors = st.slider("Number of Neighbors (K in KNN)", 1, 15, 3)
#         model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
#     elif model_choice == "Logistic Regression":
#         c_value = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
#         model = LogisticRegression(C=c_value, random_state=42)
    
#     elif model_choice == "Linear Regression":
#         model = LinearRegression()
    
#     elif model_choice == "Random Forest":
#         n_estimators = st.slider("Number of Trees", 10, 200, 100)
#         max_depth = st.slider("Max Depth of Trees", 2, 20, 10)
#         model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
#     elif model_choice == "Gradient Boosting":
#         learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
#         n_estimators = st.slider("Number of Estimators", 50, 200, 100)
#         model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=42)
    
#     elif model_choice == "SVM":
#         c_value = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
#         kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
#         model = SVC(C=c_value, kernel=kernel, probability=True, random_state=42)
    
#     elif model_choice == "XGBoost":
#         learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
#         n_estimators = st.slider("Number of Estimators", 50, 200, 100)
#         max_depth = st.slider("Max Depth of Trees", 3, 10, 5)
#         model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
#     show_dataset = st.checkbox("Show Dataset Overview", value=True)
#     show_pairplot = st.checkbox("Show Pairplot Visualization", value=False)
#     show_performance = st.checkbox("Show Model Performance", value=True)
#     show_advanced_metrics = st.checkbox("Show Advanced Metrics (F1-Score, AUC-ROC)", value=True)

# # Train the selected model
# with st.spinner('Training the model...'):
#     model.fit(X_train_poly, y_train)
# st.success('Model trained successfully!')

# # Save the model
# pickle_out = open("classifier.pkl", "wb")
# pickle.dump(model, pickle_out)
# pickle_out.close()

# # Prediction form
# with st.form("prediction_form"):
#     st.subheader("🔍 Enter Patient's Health Metrics")
#     Pregnancies = st.number_input("Pregnancies", min_value=0, value=1, step=1)
#     Glucose = st.number_input("Glucose Level", min_value=0.0, value=120.0, step=1.0)
#     BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, value=70.0, step=1.0)
#     SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0.0, value=20.0, step=1.0)
#     Insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0.0, value=80.0, step=1.0)
#     BMI = st.number_input("BMI", min_value=0.0, value=30.0, step=0.1)
#     DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5, step=0.01)
#     Age = st.number_input("Age", min_value=1, value=30, step=1)
#     submit_button = st.form_submit_button(label='🌟 Predict Diabetes')

# if submit_button:
#     if Glucose <= 0:
#         st.warning("Glucose level cannot be zero or negative.")
#     else:
#         input_data = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
#         input_data = input_data.reshape(1, -1)
#         input_data = poly.transform(scaler.transform(input_data))
#         prediction = model.predict(input_data)
        
#         if model_choice == "Linear Regression":
#             prediction = [1 if p > 0.5 else 0 for p in prediction]  # Convert probabilities to binary outcome
        
#         result = "Positive for Diabetes" if prediction[0] == 1 else "Negative for Diabetes"
#         st.success(f"🎉 Prediction: **{result}**")

# # Show dataset overview and pairplot
# if show_dataset:
#     st.markdown("### 📊 Dataset Overview & Visualization")
#     st.write(dataset.head())

# if show_pairplot:
#     st.markdown("#### 🌸 Pairplot of Diabetes Features")
#     sns.pairplot(dataset, hue="Outcome", palette="coolwarm")
#     st.pyplot(plt)

# # Model performance evaluation
# if show_performance:
#     st.markdown("### 📈 Model Performance")

#     y_pred = model.predict(X_test_poly)
#     if model_choice == "Linear Regression":
#         y_pred = [1 if p > 0.5 else 0 for p in y_pred]  # Convert probabilities to binary outcome
    
#     accuracy = accuracy_score(y_test, y_pred)
#     st.markdown(f"**Test Set Accuracy:** {accuracy:.2f}")
    
#     st.markdown("#### 📉 Confusion Matrix")
#     cm = confusion_matrix(y_test, y_pred)
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     st.pyplot(plt)

#     st.markdown("#### 📋 Classification Report")
#     st.text(classification_report(y_test, y_pred))

# # Advanced metrics: F1 Score and AUC-ROC curve
# if show_advanced_metrics:
#     st.markdown("### 🧠 Advanced Metrics")
    
#     f1 = f1_score(y_test, y_pred)
#     st.markdown(f"**F1-Score:** {f1:.2f}")
    
#     y_pred_prob = model.predict_proba(X_test_poly)[:, 1]
#     fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
#     roc_auc = auc(fpr, tpr)
    
#     roc_fig = go.Figure()
#     roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve'))
#     roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
#     roc_fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
#     st.plotly_chart(roc_fig)

#     st.markdown(f"**AUC-ROC:** {roc_auc:.2f}")

# # Model comparison section
# st.markdown("### 🤖 Model Comparison")
# models = {
#     "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
#     "Logistic Regression": LogisticRegression(C=1.0, random_state=42),
#     "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
# }

# for name, clf in models.items():
#     clf.fit(X_train_poly, y_train)
#     y_pred = clf.predict(X_test_poly)
#     accuracy = accuracy_score(y_test, y_pred)
#     st.markdown(f"**{name} Accuracy:** {accuracy:.2f}")

# # SHAP values for model explanation
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_test_poly)

# st.markdown("### 🔍 Feature Importance with SHAP")
# shap.summary_plot(shap_values, X_test_poly, plot_type="bar")
# st.pyplot(plt)

# import pandas as pd
# import numpy as np
# import streamlit as st
# import shap
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.impute import SimpleImputer
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, f1_score
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# import plotly.express as px
# import pickle
# import streamlit.components.v1 as components

# # Load dataset
# dataset = pd.read_csv("diabetes.csv")

# # Data Preprocessing
# columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
# imputer = SimpleImputer(missing_values=0, strategy='median')
# dataset[columns_to_impute] = imputer.fit_transform(dataset[columns_to_impute])

# X = dataset.drop(["Outcome"], axis=1)
# y = dataset["Outcome"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# poly = PolynomialFeatures(degree=2, interaction_only=True)
# X_train_poly = poly.fit_transform(X_train)
# X_test_poly = poly.transform(X_test)

# # Set up Streamlit App
# st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide")

# # Custom CSS for Night Theme
# night_theme = """
#     <style>
#     body {
#         background-color: #0e1117;
#         color: #cfd2d6;
#     }
#     .stApp {
#         background-color: #0e1117;
#         color: #cfd2d6;
#     }
#     h1, h2, h3, h4, h5, h6 {
#         color: #cfd2d6;
#     }
#     .stButton button {
#         background-color: #1f77b4;
#         color: #ffffff;
#     }
#     .stSidebar {
#         background-color: #161b22;
#     }
#     </style>
# """
# st.markdown(night_theme, unsafe_allow_html=True)

# st.title("🩺 Diabetes Prediction App")

# st.markdown("""
#     Welcome to the advanced version of the Diabetes Prediction App! This version allows you to choose from multiple algorithms, 
#     tune hyperparameters, and get detailed model performance insights.
# """)

# with st.sidebar:
#     st.header("🔧 Model Settings")
#     model_choice = st.selectbox("Select Model", ["K-Nearest Neighbors", "Logistic Regression", "Linear Regression", 
#                                                  "Random Forest", "Gradient Boosting", "SVM", "XGBoost"])
    
#     if model_choice == "K-Nearest Neighbors":
#         n_neighbors = st.slider("Number of Neighbors (K in KNN)", 1, 15, 3)
#         model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
#     elif model_choice == "Logistic Regression":
#         c_value = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
#         model = LogisticRegression(C=c_value, random_state=42)
    
#     elif model_choice == "Linear Regression":
#         model = LinearRegression()
    
#     elif model_choice == "Random Forest":
#         n_estimators = st.slider("Number of Trees", 10, 200, 100)
#         max_depth = st.slider("Max Depth of Trees", 2, 20, 10)
#         model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
#     elif model_choice == "Gradient Boosting":
#         learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
#         n_estimators = st.slider("Number of Estimators", 50, 200, 100)
#         model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=42)
    
#     elif model_choice == "SVM":
#         c_value = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
#         kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
#         model = SVC(C=c_value, kernel=kernel, probability=True, random_state=42)
    
#     elif model_choice == "XGBoost":
#         learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
#         n_estimators = st.slider("Number of Estimators", 50, 200, 100)
#         max_depth = st.slider("Max Depth of Trees", 3, 10, 5)
#         model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
#     show_dataset = st.checkbox("Show Dataset Overview", value=True)
#     show_pairplot = st.checkbox("Show Pairplot Visualization", value=False)
#     show_performance = st.checkbox("Show Model Performance", value=True)
#     show_advanced_metrics = st.checkbox("Show Advanced Metrics (F1-Score, AUC-ROC)", value=True)

# # Train the model
# model.fit(X_train_poly, y_train)

# # Save the model
# pickle_out = open("classifier.pkl", "wb")
# pickle.dump(model, pickle_out)
# pickle_out.close()

# with st.form("prediction_form"):
#     st.subheader("🔍 Enter Patient's Health Metrics")
#     Pregnancies = st.number_input("Pregnancies", min_value=0, value=1, step=1)
#     Glucose = st.number_input("Glucose Level", min_value=0.0, value=120.0, step=1.0)
#     BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, value=70.0, step=1.0)
#     SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0.0, value=20.0, step=1.0)
#     Insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0.0, value=80.0, step=1.0)
#     BMI = st.number_input("BMI", min_value=0.0, value=30.0, step=0.1)

# import pandas as pd
# import numpy as np
# import streamlit as st
# import shap
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.impute import SimpleImputer
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, f1_score
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# import plotly.express as px
# import pickle
# import streamlit.components.v1 as components

# # Load dataset
# dataset = pd.read_csv("diabetes.csv")

# # Data Preprocessing
# columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
# imputer = SimpleImputer(missing_values=0, strategy='median')
# dataset[columns_to_impute] = imputer.fit_transform(dataset[columns_to_impute])

# X = dataset.drop(["Outcome"], axis=1)
# y = dataset["Outcome"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# poly = PolynomialFeatures(degree=2, interaction_only=True)
# X_train_poly = poly.fit_transform(X_train)
# X_test_poly = poly.transform(X_test)

# # Set up Streamlit App
# st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide")

# # Custom CSS for Night Theme
# night_theme = """
#     <style>
#     body {
#         background-color: #0e1117;
#         color: #cfd2d6;
#     }
#     .stApp {
#         background-color: #0e1117;
#         color: #cfd2d6;
#     }
#     h1, h2, h3, h4, h5, h6 {
#         color: #cfd2d6;
#     }
#     .stButton button {
#         background-color: #1f77b4;
#         color: #ffffff;
#     }
#     .stSidebar {
#         background-color: #161b22;
#     }
#     </style>
# """
# st.markdown(night_theme, unsafe_allow_html=True)

# st.title("🩺 Diabetes Prediction App")

# st.markdown("""
#     Welcome to the advanced version of the Diabetes Prediction App! This version allows you to choose from multiple algorithms, 
#     tune hyperparameters, and get detailed model performance insights.
# """)

# with st.sidebar:
#     st.header("🔧 Model Settings")
#     model_choice = st.selectbox("Select Model", ["K-Nearest Neighbors", "Logistic Regression", "Linear Regression", 
#                                                  "Random Forest", "Gradient Boosting", "SVM", "XGBoost"])
    
#     if model_choice == "K-Nearest Neighbors":
#         n_neighbors = st.slider("Number of Neighbors (K in KNN)", 1, 15, 3)
#         model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
#     elif model_choice == "Logistic Regression":
#         c_value = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
#         model = LogisticRegression(C=c_value, random_state=42)
    
#     elif model_choice == "Linear Regression":
#         model = LinearRegression()
    
#     elif model_choice == "Random Forest":
#         n_estimators = st.slider("Number of Trees", 10, 200, 100)
#         max_depth = st.slider("Max Depth of Trees", 2, 20, 10)
#         model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
#     elif model_choice == "Gradient Boosting":
#         learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
#         n_estimators = st.slider("Number of Estimators", 50, 200, 100)
#         model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=42)
    
#     elif model_choice == "SVM":
#         c_value = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
#         kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
#         model = SVC(C=c_value, kernel=kernel, probability=True, random_state=42)
    
#     elif model_choice == "XGBoost":
#         learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
#         n_estimators = st.slider("Number of Estimators", 50, 200, 100)
#         max_depth = st.slider("Max Depth of Trees", 3, 10, 5)
#         model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
#     show_dataset = st.checkbox("Show Dataset Overview", value=True)
#     show_pairplot = st.checkbox("Show Pairplot Visualization", value=False)
#     show_performance = st.checkbox("Show Model Performance", value=True)
#     show_advanced_metrics = st.checkbox("Show Advanced Metrics (F1-Score, AUC-ROC)", value=True)

# # Train the model
# model.fit(X_train_poly, y_train)

# # Save the model
# pickle_out = open("classifier.pkl", "wb")
# pickle.dump(model, pickle_out)
# pickle_out.close()

# with st.form("prediction_form"):
#     st.subheader("🔍 Enter Patient's Health Metrics")
#     Pregnancies = st.number_input("Pregnancies", min_value=0, value=1, step=1)
#     Glucose = st.number_input("Glucose Level", min_value=0.0, value=120.0, step=1.0)
#     BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, value=70.0, step=1.0)
#     SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0.0, value=20.0, step=1.0)
#     Insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0.0, value=80.0, step=1.0)
#     BMI = st.number_input("BMI", min_value=0.0, value=30.0, step=0.1)
#     DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5, step=0.01)
#     Age = st.number_input("Age", min_value=1, value=30, step=1)
#     submit_button = st.form_submit_button(label='🌟 Predict Diabetes')

# if submit_button:
#     input_data = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
#     input_data = input_data.reshape(1, -1)
#     input_data = poly.transform(scaler.transform(input_data))
#     prediction = model.predict(input_data)
    
#     if model_choice == "Linear Regression":
#         prediction = [1 if p > 0.5 else 0 for p in prediction]  # Convert probabilities to binary outcome
    
#     result = "Positive for Diabetes" if prediction[0] == 1 else "Negative for Diabetes"
#     st.success(f"🎉 Prediction: **{result}**")

# # Display Dataset
# if show_dataset:
#     st.markdown("### 📊 Dataset Overview & Visualization")
#     st.write(dataset.head())

# # Pairplot Visualization
# if show_pairplot:
#     st.markdown("#### 🌸 Pairplot of Diabetes Features")
#     sns.pairplot(dataset, hue="Outcome", palette="coolwarm")
#     st.pyplot(plt)

# # Model Performance
# if show_performance:
#     st.markdown("### 📈 Model Performance")

#     y_pred = model.predict(X_test_poly)
#     if model_choice == "Linear Regression":
#         y_pred = [1 if p > 0.5 else 0 for p in y_pred]  # Convert probabilities to binary outcome
    
#     accuracy = accuracy_score(y_test, y_pred)
#     st.markdown(f"**Test Set Accuracy:** {accuracy:.2f}")
    
#     st.markdown("#### 📉 Confusion Matrix")
#     cm = confusion_matrix(y_test, y_pred)
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     st.pyplot(plt)

#     st.markdown("#### 📋 Classification Report")
#     st.text(classification_report(y_test, y_pred))

# # Advanced metrics: F1 Score and AUC-ROC
# if show_advanced_metrics:
#     st.markdown("### 📊 Advanced Metrics")

#     # F1 Score
#     f1 = f1_score(y_test, y_pred)
#     st.markdown(f"**F1 Score:** {f1:.2f}")
    
#     # AUC-ROC Curve
#     if hasattr(model, "predict_proba"):
#         y_pred_prob = model.predict_proba(X_test_poly)[:, 1]
#     else:
#         y_pred_prob = (model.predict(X_test_poly) - model.predict(X_test_poly).min()) / (model.predict(X_test_poly).max() - model.predict(X_test_poly).min()) # Normalize output

#     fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
#     roc_auc = auc(fpr, tpr)
#     st.markdown(f"**AUC-ROC:** {roc_auc:.2f}")

#     fig, ax = plt.subplots()
#     ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
#     ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     ax.set_xlim([0.0, 1.0])
#     ax.set_ylim([0.0, 1.05])
#     ax.set_xlabel('False Positive Rate')
#     ax.set_ylabel('True Positive Rate')
#     ax.set_title('Receiver Operating Characteristic')
#     ax.legend(loc="lower right")
#     st.pyplot(fig)

# # SHAP Explainer and Summary Plot
# st.markdown("### 🌟 SHAP Visualizations")

# # Only show SHAP explanations for models that support them
# if model_choice in ["Random Forest", "Gradient Boosting", "XGBoost"]:
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X_test_poly)

#     # Summary plot for SHAP values
#     st.markdown("#### Summary Plot")
#     fig, ax = plt.subplots()
#     shap.summary_plot(shap_values, X_test_poly, feature_names=poly.get_feature_names_out(X.columns), show=False)
#     st.pyplot(fig)

#     # Feature importance based on SHAP values
#     st.markdown("#### Feature Importance")
#     fig, ax = plt.subplots()
#     shap.summary_plot(shap_values, X_test_poly, feature_names=poly.get_feature_names_out(X.columns), plot_type="bar", show=False)
#     st.pyplot(fig)

# elif model_choice in ["Logistic Regression", "Linear Regression", "SVM"]:
#     st.markdown("SHAP explanations are not supported for this model choice.")
# else:
#     st.markdown("SHAP explanations are only available for tree-based models.")




# import pandas as pd
# import numpy as np
# import streamlit as st
# import shap
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.impute import SimpleImputer
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, f1_score
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# import plotly.express as px
# import pickle
# import streamlit.components.v1 as components

# # Load dataset
# dataset = pd.read_csv("diabetes.csv")

# # Data Preprocessing
# columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
# imputer = SimpleImputer(missing_values=0, strategy='median')
# dataset[columns_to_impute] = imputer.fit_transform(dataset[columns_to_impute])

# X = dataset.drop(["Outcome"], axis=1)
# y = dataset["Outcome"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# poly = PolynomialFeatures(degree=2, interaction_only=True)
# X_train_poly = poly.fit_transform(X_train)
# X_test_poly = poly.transform(X_test)

# # Set up Streamlit App
# st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide")

# # Custom CSS for Night Theme
# night_theme = """
#     <style>
#     body {
#         background-color: #0e1117;
#         color: #cfd2d6;
#     }
#     .stApp {
#         background-color: #0e1117;
#         color: #cfd2d6;
#     }
#     h1, h2, h3, h4, h5, h6 {
#         color: #cfd2d6;
#     }
#     .stButton button {
#         background-color: #1f77b4;
#         color: #ffffff;
#     }
#     .stSidebar {
#         background-color: #161b22;
#     }
#     </style>
# """
# st.markdown(night_theme, unsafe_allow_html=True)

# st.title("🩺 Diabetes Prediction App")

# st.markdown("""
#     Welcome to the advanced version of the Diabetes Prediction App! This version allows you to choose from multiple algorithms, 
#     tune hyperparameters, and get detailed model performance insights.
# """)

# with st.sidebar:
#     st.header("🔧 Model Settings")
#     model_choice = st.selectbox("Select Model", ["K-Nearest Neighbors", "Logistic Regression", "Linear Regression", 
#                                                  "Random Forest", "Gradient Boosting", "SVM", "XGBoost"])
    
#     if model_choice == "K-Nearest Neighbors":
#         n_neighbors = st.slider("Number of Neighbors (K in KNN)", 1, 15, 3)
#         model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
#     elif model_choice == "Logistic Regression":
#         c_value = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
#         model = LogisticRegression(C=c_value, random_state=42)
    
#     elif model_choice == "Linear Regression":
#         model = LinearRegression()
    
#     elif model_choice == "Random Forest":
#         n_estimators = st.slider("Number of Trees", 10, 200, 100)
#         max_depth = st.slider("Max Depth of Trees", 2, 20, 10)
#         model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
#     elif model_choice == "Gradient Boosting":
#         learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
#         n_estimators = st.slider("Number of Estimators", 50, 200, 100)
#         model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=42)
    
#     elif model_choice == "SVM":
#         c_value = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
#         kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
#         model = SVC(C=c_value, kernel=kernel, probability=True, random_state=42)
    
#     elif model_choice == "XGBoost":
#         learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
#         n_estimators = st.slider("Number of Estimators", 50, 200, 100)
#         max_depth = st.slider("Max Depth of Trees", 3, 10, 5)
#         model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
#     show_dataset = st.checkbox("Show Dataset Overview", value=True)
#     show_pairplot = st.checkbox("Show Pairplot Visualization", value=False)
#     show_performance = st.checkbox("Show Model Performance", value=True)
#     show_advanced_metrics = st.checkbox("Show Advanced Metrics (F1-Score, AUC-ROC)", value=True)

# # Train the model
# model.fit(X_train_poly, y_train)

# # Save the model
# pickle_out = open("classifier.pkl", "wb")
# pickle.dump(model, pickle_out)
# pickle_out.close()

# with st.form("prediction_form"):
#     st.subheader("🔍 Enter Patient's Health Metrics")
#     Pregnancies = st.number_input("Pregnancies", min_value=0, value=1, step=1)
#     Glucose = st.number_input("Glucose Level", min_value=0.0, value=120.0, step=1.0)
#     BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, value=70.0, step=1.0)
#     SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0.0, value=20.0, step=1.0)
#     Insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0.0, value=80.0, step=1.0)
#     BMI = st.number_input("BMI", min_value=0.0, value=30.0, step=0.1)
#     DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5, step=0.01)
#     Age = st.number_input("Age", min_value=1, value=30, step=1)
#     submit_button = st.form_submit_button(label='🌟 Predict Diabetes')

# if submit_button:
#     input_data = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
#     input_data = input_data.reshape(1, -1)
#     input_data = poly.transform(scaler.transform(input_data))
#     prediction = model.predict(input_data)
    
#     if model_choice == "Linear Regression":
#         prediction = [1 if p > 0.5 else 0 for p in prediction]  # Convert probabilities to binary outcome
    
#     result = "Positive for Diabetes" if prediction[0] == 1 else "Negative for Diabetes"
#     st.success(f"🎉 Prediction: **{result}**")

# # Display Dataset
# if show_dataset:
#     st.markdown("### 📊 Dataset Overview & Visualization")
#     st.write(dataset.head())

# # Pairplot Visualization
# if show_pairplot:
#     st.markdown("#### 🌸 Pairplot of Diabetes Features")
#     sns.pairplot(dataset, hue="Outcome", palette="coolwarm")
#     st.pyplot(plt)

# # Model Performance
# if show_performance:
#     st.markdown("### 📈 Model Performance")

#     y_pred = model.predict(X_test_poly)
#     if model_choice == "Linear Regression":
#         y_pred = [1 if p > 0.5 else 0 for p in y_pred]  # Convert probabilities to binary outcome
    
#     accuracy = accuracy_score(y_test, y_pred)
#     st.markdown(f"**Test Set Accuracy:** {accuracy:.2f}")
    
#     st.markdown("#### 📉 Confusion Matrix")
#     cm = confusion_matrix(y_test, y_pred)
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     st.pyplot(plt)

#     st.markdown("#### 📋 Classification Report")
#     st.text(classification_report(y_test, y_pred))

# # Advanced metrics: F1 Score and AUC-ROC
# if show_advanced_metrics:
#     st.markdown("### 📊 Advanced Metrics")

#     # F1 Score
#     f1 = f1_score(y_test, y_pred)
#     st.markdown(f"**F1 Score:** {f1:.2f}")
    
#     # AUC-ROC Curve
#     if hasattr(model, "predict_proba"):
#         y_pred_prob = model.predict_proba(X_test_poly)[:, 1]
#     else:
#         y_pred_prob = (model.predict(X_test_poly) - model.predict(X_test_poly).min()) / \
#                       (model.predict(X_test_poly).max() - model.predict(X_test_poly).min())
    
#     fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
#     auc_score = auc(fpr, tpr)

#     st.markdown(f"**AUC-ROC Score:** {auc_score:.2f}")

#     fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode='lines', name='AUC-ROC Curve'))
#     fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='grey')))
#     fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
#                       xaxis_title='False Positive Rate',
#                       yaxis_title='True Positive Rate',
#                       showlegend=False)
#     st.plotly_chart(fig)

# # # SHAP values for feature importance
# # st.markdown("### 🌟 SHAP Feature Importance")
# # explainer = shap.KernelExplainer(model.predict, X_train_poly[:100])
# # shap_values = explainer.shap_values(X_test_poly[:10])

# # # Since SHAP values are calculated on the transformed data, pass the transformed test data for plotting
# # shap.summary_plot(shap_values, X_test_poly[:10], feature_names=poly.get_feature_names_out())

# # # Load JS code for the SHAP summary plot
# # shap_html = f"<script>{shap.getjs()}</script>"
# # components.html(shap_html, height=700)

# # SHAP Explainer and Summary Plot
# st.markdown("### 🌟 SHAP Visualizations")

# # Only show SHAP explanations for models that support them
# if model_choice in ["Random Forest", "Gradient Boosting", "XGBoost"]:
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X_test_poly)

#     # Summary plot for SHAP values
#     st.markdown("#### Summary Plot")
#     fig, ax = plt.subplots()
#     shap.summary_plot(shap_values, X_test_poly, feature_names=poly.get_feature_names_out(X.columns), show=False)
#     st.pyplot(fig)

#     # Feature importance based on SHAP values
#     st.markdown("#### Feature Importance")
#     fig, ax = plt.subplots()
#     shap.summary_plot(shap_values, X_test_poly, feature_names=poly.get_feature_names_out(X.columns), plot_type="bar", show=False)
#     st.pyplot(fig)

# elif model_choice in ["Logistic Regression", "Linear Regression", "SVM"]:
#     st.markdown("SHAP explanations are not supported for this model choice.")
# else:
#     st.markdown("SHAP explanations are only available for tree-based models.")




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve, auc
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import shap
import pickle
import requests
import streamlit_lottie as st_lottie

# Load the dataset
dataset = pd.read_csv("diabetes.csv")

# Impute missing values
columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
imputer = SimpleImputer(missing_values=0, strategy='median')
dataset[columns_to_impute] = imputer.fit_transform(dataset[columns_to_impute])

# Split the dataset
X = dataset.drop(["Outcome"], axis=1)
y = dataset["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Apply Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Set Streamlit page configuration
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide")

# Custom CSS for Night Theme
night_theme = """
    <style>
    body {
        background-color: #0e1117;
        color: #cfd2d6;
    }
    .stApp {
        background-color: #0e1117;
        color: #cfd2d6;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #cfd2d6;
    }
    .stButton button {
        background-color: #1f77b4;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #161b22;
    }
    </style>
"""
st.markdown(night_theme, unsafe_allow_html=True)

# Load Lottie animation
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# lottie_diabetes = load_lottie_url("https://assets8.lottiefiles.com/packages/lf20_jcikwtux.json")
# Load a Lottie animation using the URL
# lottie_diabetes = load_lottie_url("https://lottie.host/cb54b283-5df4-4a94-b096-f20609d6cedd/OieGG3bmfC.json")
# lottie_diabetes = load_lottie_url("https://lottie.host/c143e0d5-835f-4679-9c06-d7cbc34776a6/xvn6XvaIF1.json")
# lottie_diabetes = load_lottie_url("https://assets8.lottiefiles.com/packages/lf20_jcikwtux.json")
# lottie_diabetes = load_lottie_url("https://lottie.host/c7a1c31d-cc5f-454c-96e5-a76848935dfc/T76RieKTWP.json")
lottie_diabetes = load_lottie_url("https://lottie.host/43d3d0fb-9f7e-46dc-9c18-355cfd7836f1/dB6JnNNwE2.json")
# lottie_diabetes = load_lottie_url("https://lottie.host/7452016b-138a-4482-9e6a-52743a98c03b/DRqOQID6ht.json")
# lottie_diabetes = load_lottie_url("")
# lottie_diabetes = load_lottie_url("")


st.title("🩺 Diabetes Prediction App")
st.markdown("""
    Welcome to the advanced version of the Diabetes Prediction App! This version allows you to choose from multiple algorithms, 
    tune hyperparameters, and get detailed model performance insights.
""")

# Sidebar for model selection and settings
with st.sidebar:
    st.header("🔧 Model Settings")
    st_lottie.st_lottie(lottie_diabetes, height=200, width=200)

    model_choice = st.selectbox("Select Model", ["K-Nearest Neighbors", "Logistic Regression", "Linear Regression", 
                                                 "Random Forest", "Gradient Boosting", "SVM", "XGBoost"])
    
    if model_choice == "K-Nearest Neighbors":
        n_neighbors = st.slider("Number of Neighbors (K in KNN)", 1, 15, 3)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    elif model_choice == "Logistic Regression":
        c_value = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
        model = LogisticRegression(C=c_value, random_state=42)
    
    elif model_choice == "Linear Regression":
        model = LinearRegression()
    
    elif model_choice == "Random Forest":
        n_estimators = st.slider("Number of Trees", 10, 200, 100)
        max_depth = st.slider("Max Depth of Trees", 2, 20, 10)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
    elif model_choice == "Gradient Boosting":
        learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
        n_estimators = st.slider("Number of Estimators", 50, 200, 100)
        model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=42)
    
    elif model_choice == "SVM":
        c_value = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
        kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
        model = SVC(C=c_value, kernel=kernel, probability=True, random_state=42)
    
    elif model_choice == "XGBoost":
        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
        n_estimators = st.slider("Number of Estimators", 50, 200, 100)
        max_depth = st.slider("Max Depth of Trees", 3, 10, 5)
        model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
    show_dataset = st.checkbox("Show Dataset Overview", value=True)
    show_pairplot = st.checkbox("Show Pairplot Visualization", value=False)
    show_performance = st.checkbox("Show Model Performance", value=True)
    show_advanced_metrics = st.checkbox("Show Advanced Metrics (F1-Score, AUC-ROC)", value=True)
    
    # Option to Show SHAP Visualizations
    show_shap = st.checkbox("Show SHAP Visualizations", value=False)

# Train the selected model
with st.spinner('Training the model...'):
    model.fit(X_train_poly, y_train)
# st.success('Model trained successfully!')

# Save the model
pickle_out = open("classifier.pkl", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()

# Prediction form
with st.form("prediction_form"):
    st.subheader("🔍 Enter Patient's Health Metrics")
    Pregnancies = st.number_input("Pregnancies", min_value=0, value=1, step=1)
    Glucose = st.number_input("Glucose Level", min_value=0.0, value=120.0, step=1.0)
    BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, value=70.0, step=1.0)
    SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0.0, value=20.0, step=1.0)
    Insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0.0, value=80.0, step=1.0)
    BMI = st.number_input("BMI", min_value=0.0, value=30.0, step=0.1)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5, step=0.01)
    Age = st.number_input("Age", min_value=1, value=30, step=1)
    submit_button = st.form_submit_button(label='🌟 Predict Diabetes')

if submit_button:
    if Glucose <= 0:
        st.warning("Glucose level cannot be zero or negative.")
    else:
        input_data = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        input_data = input_data.reshape(1, -1)
        input_data = poly.transform(scaler.transform(input_data))
        prediction = model.predict(input_data)
        
        if model_choice == "Linear Regression":
            prediction = [1 if p > 0.5 else 0 for p in prediction]  
        
        result = "Positive for Diabetes" if prediction[0] == 1 else "Negative for Diabetes"
        st.success(f"🎉 Prediction: **{result}**")

# Show dataset overview and pairplot
if show_dataset:
    st.markdown("### 📊 Dataset Overview & Visualization")
    st.write(dataset.head())

if show_pairplot:
    st.markdown("#### 🌸 Pairplot of Diabetes Features")
    sns.pairplot(dataset, hue="Outcome", palette="coolwarm")
    st.pyplot(plt)

# Model performance evaluation
if show_performance:
    st.markdown("### 📈 Model Performance")

    y_pred = model.predict(X_test_poly)
    if model_choice == "Linear Regression":
        y_pred = [1 if p > 0.5 else 0 for p in y_pred]  # Convert probabilities to binary outcome

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write("**Classification Report:**")
    st.json(report)  # This will show the report in a more structured format

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))  # Adjust figure size to fit within the Streamlit layout
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)  # Display the plot

    # Optionally use tight_layout if needed
    # fig.tight_layout()

# Advanced metrics: F1 Score and AUC-ROC
if show_advanced_metrics:
    st.markdown("### 📊 Advanced Metrics")

    # F1 Score
    f1 = f1_score(y_test, y_pred)
    st.markdown(f"**F1 Score:** {f1:.2f}")
    
    # AUC-ROC Curve
    if hasattr(model, "predict_proba"):
        y_pred_prob = model.predict_proba(X_test_poly)[:, 1]
    else:
        y_pred_prob = (model.predict(X_test_poly) - model.predict(X_test_poly).min()) / \
                      (model.predict(X_test_poly).max() - model.predict(X_test_poly).min())
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_score = auc(fpr, tpr)

    st.markdown(f"**AUC-ROC Score:** {auc_score:.2f}")

    fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode='lines', name='AUC-ROC Curve'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='grey')))
    fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate',
                      showlegend=False)
    st.plotly_chart(fig)

# st.markdown("### 🌟 SHAP Visualizations")

# # Only show SHAP explanations for models that support them
# if model_choice in ["Random Forest", "Gradient Boosting", "XGBoost"]:
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X_test_poly)

#     # Summary plot for SHAP values
#     st.markdown("#### Summary Plot")
#     fig, ax = plt.subplots()
#     shap.summary_plot(shap_values, X_test_poly, feature_names=poly.get_feature_names_out(X.columns), show=False)
#     st.pyplot(fig)

#     # Feature importance based on SHAP values
#     st.markdown("#### Feature Importance")
#     fig, ax = plt.subplots()
#     shap.summary_plot(shap_values, X_test_poly, feature_names=poly.get_feature_names_out(X.columns), plot_type="bar", show=False)
#     st.pyplot(fig)

# elif model_choice in ["Logistic Regression", "Linear Regression", "SVM"]:
#     st.markdown("SHAP explanations are not supported for this model choice.")
# else:
#     st.markdown("SHAP explanations are only available for tree-based models.")

# SHAP Visualizations
if show_shap:
    st.markdown("### 🌟 SHAP Visualizations")

    # Only show SHAP explanations for models that support them
    if model_choice in ["Random Forest", "Gradient Boosting", "XGBoost"]:
        try:
            # Create the SHAP explainer
            explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
            # Compute SHAP values
            shap_values = explainer.shap_values(X_test_poly)
            
            # Summary plot for SHAP values
            st.markdown("#### Summary Plot")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_test_poly, feature_names=poly.get_feature_names_out(X.columns), show=False)
            st.pyplot(fig)

            # Feature importance based on SHAP values
            st.markdown("#### Feature Importance")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_test_poly, feature_names=poly.get_feature_names_out(X.columns), plot_type="bar", show=False)
            st.pyplot(fig)

        except Exception as e:  # Catch general exceptions
            st.error(f"An error occurred with SHAP visualizations: {e}")
            st.write("Try adjusting the model or data to resolve the issue.")
            
    else:
        st.markdown("SHAP explanations are only available for tree-based models.")



# End of the app
st.markdown("Thank you for using the **Diabetes Prediction App**! Stay healthy and take care.")
