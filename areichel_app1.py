# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set up the title and description of the app
st.title('Fetal Health Classification: A Machine Learning App')
# Display an image of penguins
st.image('fetal_health_image.gif', width = 600)
st.write('Utilize our advanced Machine Learning Application to predict fetal health classifications.')

# Reading the pickle file that we created before  
with open('decision_tree_fetal.pickle', 'rb') as dt_pickle:
    clf = pickle.load(dt_pickle)
with open('random_forest_fetal.pickle', 'rb') as rf_pickle:
    clf1 = pickle.load(rf_pickle)
with open('Adaboost_fetal.pickle', 'rb') as ab_pickle:
    clf2 = pickle.load(ab_pickle)
with open('softvoting_fetal.pickle', 'rb') as sv_pickle:
    clf3 = pickle.load(sv_pickle)

df = pd.read_csv('fetal_health.csv').drop(columns='fetal_health')


# Sidebar
st.sidebar.header('Fetal Health Features Input')

# Upload CSV section
uploaded_file = st.sidebar.file_uploader("Upload your data", type="csv")
st.sidebar.warning('⚠️ Ensure your data upload follows the format displayed below')
st.sidebar.dataframe(df.head(5))

# Select Model for Prediction
model_selection = st.sidebar.radio("Choose Model for  Prediction", options=['Random Forest','Decision Tree', 'AdaBoost', 'Soft Voting'])
st.sidebar.info(f"You selected: {model_selection}")

if uploaded_file is not None:
    st.success('✅ *CSV file uploaded successfully*')
    if model_selection == 'Decision Tree':
         model = clf
    elif model_selection == 'Random Forest':
        model = clf1
    elif model_selection == 'AdaBoost':
         model = clf2
    elif model_selection == 'Soft Voting':
         model = clf3
 
    st.subheader(f"Predicting Fetal Health Class Using {model_selection} Model")
    input = pd.read_csv(uploaded_file)
    features = input
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    
    # Class Prediction and Coloring
    prediction_name = []
    for i in prediction:
        if i == 1:
            prediction_name.append("Normal Class")
        elif i == 2:
            prediction_name.append("Suspect Class")
        elif i == 3:
            prediction_name.append("Pathological Class") 
    input['Predicted Fetal Health'] = prediction_name
    def colors(prediction_name):
         if prediction_name == "Normal Class":
              return 'background-color: Lime'
         elif prediction_name == "Suspect Class":
            return 'background-color: Yellow'
         elif prediction_name == "Pathological Class":
              return 'background-color: Orange'
    colored_input = input.style.applymap(colors, subset=['Predicted Fetal Health'])  
    
    # Prediction Probability
    prediction_probs = []
    for i in probability:
        max_prob = i.max()
        prediction_probs.append(f"{max_prob * 100:.2f}") 
    input['Prediction Probability (%)'] = prediction_probs
    st.dataframe(colored_input)
    
    # Showing additional items in tabs
    st.subheader("Model Performance and Insights")
    tab1, tab2, tab3, = st.tabs(["Confusion Matrix", "Classification Report", "Feature Importance"])
    # Tab 1: Confusion Matrix
    with tab1:
        st.write("### Confusion Matrix")
        if model_selection == 'Decision Tree':
            st.image('con_mat_dt.svg')
        elif model_selection == "Random Forest":
            st.image('con_mat_rf.svg')
        elif model_selection == "AdaBoost":
            st.image('con_mat_ab.svg')
        elif model_selection == "Soft Voting":
            st.image('con_mat_sv.svg')
        st.caption("Confusion Matrix of model predictions.")

    # Tab 2: Classification Report
    with tab2:
        st.write("### Classification Report")
        if model_selection == 'Decision Tree':
            report_df = pd.read_csv('class_report_dt.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))  
        elif model_selection == "Random Forest":
            report_df = pd.read_csv('class_report_rf.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
        elif model_selection == "AdaBoost":
            report_df = pd.read_csv('class_report_ab.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
        elif model_selection == "Soft Voting":
            report_df = pd.read_csv('class_report_sv.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
        st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")

    # Tab 3: Feature Importance Visualization
    with tab3:
        st.write("### Feature Importance")
        if model_selection == 'Decision Tree':
            st.image('feature_imp_dt.svg')
        elif model_selection == "Random Forest":
            st.image('feature_imp_rf.svg')
        elif model_selection == "AdaBoost":
            st.image('feature_imp_ab.svg')
        elif model_selection == "Soft Voting":
            st.image('feature_imp_sv.svg')
        st.caption("Features used in this prediction are ranked by relative importance.")
else:
    st.info('*Please upload data to proceed*')


