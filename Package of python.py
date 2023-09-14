import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from sklearn.impute import SimpleImputer
from pycaret.classification import *

st.header('Package of ML')

st.subheader('Load the data')
data=st.file_uploader("Upload Data CSV Or xlsx")

if data is not None :
    if str(data.name).endswith("xls") or str(data.name).endswith("xlsx"):
        data=pd.read_excel(data,encoding='latin-1')
        print(data.head(10))
        st.write(data.head(10))
    elif str(data.name).endswith("csv"):
        data=pd.read_csv(data,encoding='latin-1')
        print(data.head(10))
        st.write(data.head(10))
    else :
            st.info("Please upload a csv or xlsx file")
    	


# st.header("Perform EDA")
# st.text("Handle null values")
#Detect column types:
if data is not None:
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

#Handle null values

# # For numeric columns, replace null values with the mean
# numeric_imputer = SimpleImputer(strategy='mean')
# data[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])

# # For categorical columns, replace null values with the most frequent value
# categorical_imputer = SimpleImputer(strategy='most_frequent')
# data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])
# st.write(data)


st.text('Drop any column you want')

if data is not None :
    list_of_data=data.columns

    colsLinea = st.selectbox("Choics Column",
                    list_of_data)
    btn=st.button('Drop this Column')
    if btn:
        data=data.drop(colsLinea,axis=1)

st.text('Target Column')
if data is not None: 
    Target_Column= st.selectbox("Choics the column that you want to predict",
                    list_of_data)

# Detect column type
    column_type = data[Target_Column].dtype

# Determine task type based on column type
    if pd.api.types.is_numeric_dtype(column_type):
        task_type = "regression"
        print("re")
    else:
        task_type = "classification"
        print("cl")

if data is not None: 
    Technique=["Mean","Median", "Mode"]
    category_list=["Most frequent","add Additional class" ]
    if task_type=="regression":
            Technique_type= st.selectbox("Choics the Technique that you want to apply",
                    Technique)
            if Technique_type=="Mean":
                data[Target_Column].fillna(data[Target_Column].mean(), inplace=True)
                st.write(data)
            elif Technique_type=="Median":
                data[Target_Column].fillna(data[Target_Column].median(), inplace=True)
                st.write(data)
            elif Technique_type=="Mode":
                data[Target_Column].fillna(data[Target_Column].mode(), inplace=True)
                st.write(data)
    elif task_type=="classification":
            Technique_type= st.selectbox("Choics the Technique that you want to apply",
                    category_list)
            if Technique_type=="Most frequent":
                data[Target_Column].fillna(data[Target_Column].mode(), inplace=True)
                st.write(data)
            elif Technique_type=="add Additional class":
                data[Target_Column].fillna("Unknown", inplace=True)
                st.write(data)

    clf = setup(data=data, target=Target_Column)

    best_model = compare_models()
    final_report = pull()
    st.write("pycaret algorithms")
    print(final_report)
    st.write(final_report)
