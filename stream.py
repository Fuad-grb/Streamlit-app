# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 22:15:14 2024

@author: Fuad Gurbanov
"""
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
# pip install seaborn
# pip install matplotlib.pyplot
import seaborn as sns
import matplotlib.pyplot as plt
icon = Image.open(r'istockphoto-1407983911-612x612.jpg')
banner = Image.open(
    r'pngtree-d-render-of-a-user-friendly-seo-data-analytics-interface-with-image_3633760.jpg')
sidebar = Image.open(r'images (1).jpg')


st.set_page_config(page_title='AnalyticaView', page_icon=icon, layout='wide')
st.sidebar.image(sidebar.resize((320, 180)))
st.image(banner.resize((800, 360)))
st.sidebar.markdown("""
      <style>
       .st-emotion-cache-vk3wp9{
         background-color: #FFBF00;
        padding: 0 0 0 2rem;
       }
        .st-emotion-cache-1629p8f h2{
            font-size: 34px;
        }
      
   </>""", unsafe_allow_html=True)
st.title('Quantalytic')
st.header('Streamlined Data Insights: Unleashing the Power of Simple Data Science for User Comfort.')
st.subheader('Explore and Analyze Your Data')
st.markdown(
    """
    <div style='background-color: #4682B4; padding: 10px; border-radius: 10px;'>
        <h3 style='color: white; font-weight: bold;'>About this code:</h3>
        <p style='color: white;'>This Streamlit app provides functionalities for loading datasets, exploring data, 
        performing exploratory data analysis (EDA), and building machine learning models. Customize and extend 
        the code to meet your specific needs.
        You can utilize a pre-existing dataset in the application or upload your own dataset. </p>
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio("Menu", ('Load Dataset', 'Explore Ready Dataset'))
st.markdown("""
      <style>
       .st-emotion-cache-vk3wp9{
         background-color: #FFBF00;
        padding: 0 0 0 2rem;
       }
        .st-emotion-cache-1629p8f h2{
            font-size: 34px;
        }
      
   </>""", unsafe_allow_html=True)


def show_data(raw_data):
    first = st.checkbox('Data Glance', ('Show Data', 'Show Shape'))
    if first:
        if 'Show Data':
            st.subheader('Raw Data')
            st.dataframe(raw_data)

        if 'Show Shape':
            st.subheader('Data Shape')
            st.write(f"Number of rows: {raw_data.shape[0]}")
            st.write(f"Number of columns: {raw_data.shape[1]}")

# show unique_values


def unique_values(raw_data):
    st.dataframe(pd.DataFrame(raw_data.nunique(),
                 columns=['Number of Unique Values']))
# Describe Data


def describe(raw_data):
    if st.checkbox('Describe'):
        st.title('Describe of Data')
        raw_describe = raw_data.describe()
        st.dataframe(raw_describe)
    if st.checkbox('Show Null Values'):
        st.title('Null Values')
        null_values = pd.DataFrame(raw_data.isnull().sum(), columns=[
                                   'Sum of Null Values'])
        st.dataframe(null_values)
    if st.checkbox('Show number of unique values'):
        unique_values(raw_data)

# Data Graph


def data_graph(raw_data):
    graph_name = ['Pairplot', 'Box Plot',
                  'Scatter Plot', 'Line Plot', 'Violin Plot']
    # selected_graphs=st.selectbox(graph_name)
    selected_graphs = st.multiselect("Select Graphs", graph_name)
    # selected_graphs = [st.selectbox(name) for name in graph_name]
    if any(selected_graphs):
        numeric_data = raw_data.select_dtypes(exclude='object')
        cat_data = raw_data.select_dtypes('object')
        for i, graph in enumerate(selected_graphs):
            if selected_graphs[i]:
                st.title(graph)
                st.write(f"This is a {graph}.")

                if graph == 'Pairplot':
                    st.write(
                        "Pairplot is a great way to visualize relationships between numerical variables.")
                    a = sns.pairplot(raw_data)
                    st.pyplot(a)

                elif graph == 'Box Plot':
                    st.write(
                        "Box plots show the distribution of numerical data and identify outliers.")
                    selected_column = st.selectbox(
                        "Select a column for Box Plot", numeric_data.columns)
                    fig, axes = plt.subplots(figsize=(6, 6))
                    sns.boxplot(x=selected_column, data=numeric_data)
                    st.pyplot(fig)

                elif graph == 'Scatter Plot':
                    st.write(
                        "Scatter plots show the relationship between two numerical variables.")
                    x_column = st.selectbox(
                        "Select X-axis column", numeric_data.columns)
                    y_column = st.selectbox(
                        "Select Y-axis column", numeric_data.columns)
                    hue_options = [None] + list(cat_data.columns)
                    hue_column = st.selectbox(
                        "Select Hue column (optional)", hue_options)
                    fig, axes = plt.subplots(figsize=(6, 6))
                    sns.scatterplot(x=x_column, y=y_column,
                                    hue=hue_column, data=raw_data)
                    st.pyplot(fig)

                elif graph == 'Line Plot':
                    st.write(
                        "Line plots show the relationship between two numerical variables over a continuous interval.")
                    x_column = st.selectbox(
                        "Select X-axis column", numeric_data.columns)
                    y_column = st.selectbox(
                        "Select Y-axis column", numeric_data.columns)
                    hue_options = [None] + list(cat_data.columns)
                    hue_column = st.selectbox(
                        "Select Hue column (optional)", hue_options)
                    fig, axes = plt.subplots(figsize=(6, 6))
                    sns.lineplot(x=x_column, y=y_column,
                                 hue=hue_column, data=raw_data)
                    st.pyplot(fig)

                elif graph == 'Violin Plot':
                    st.write(
                        "Violin plots show the distribution of numerical data for different categories.")
                    # x_column = st.selectbox("Select X-axis column", numeric_data.columns)
                    y_column = st.selectbox(
                        "Select Y-axis column", numeric_data.columns)
                    hue_options = [None] + list(cat_data.columns)
                    hue_column = st.selectbox(
                        "Select Hue column (optional)", hue_options)
                    fig, axes = plt.subplots(figsize=(6, 6))
                    sns.violinplot(x=hue_column, y=y_column, data=raw_data)
                    st.pyplot(fig)
                else:
                    pass
# eda Function


def eda(raw_data):
    st.title('EDA')
    page1 = st.selectbox("Eda", ('Data Describe', 'Data Graph'))
    if page1 == 'Data Describe':
        describe(raw_data)
    if page1 == 'Data Graph':
        data_graph(raw_data)

# BUILDING MODEL
# iMPUTER
# SIMPLE IMPUTER


def simple_imputer(raw_data):
    numeric_data = raw_data.select_dtypes(include='number')
    categorical_data = raw_data.select_dtypes(exclude='number')

    numeric_strategy = st.radio('Choose Numeric Strategy', ['median', 'mean'])
    categorical_strategy = st.radio(
        'Choose Categorical Strategy', ['most_frequent'])

    from sklearn.impute import SimpleImputer

    # Impute numeric columns
    if not numeric_data.empty:
        numeric_imputer = SimpleImputer(strategy=numeric_strategy)
        numeric_imputed = pd.DataFrame(numeric_imputer.fit_transform(
            numeric_data), columns=numeric_data.columns)

    # Impute categorical columns
    if not categorical_data.empty:
        categorical_imputer = SimpleImputer(
            strategy=categorical_strategy, fill_value='missing')
        categorical_imputed = pd.DataFrame(categorical_imputer.fit_transform(
            categorical_data), columns=categorical_data.columns)

    # Concatenate the imputed dataframes
    if not numeric_data.empty and not categorical_data.empty:
        new_data = pd.concat([numeric_imputed, categorical_imputed], axis=1)
    elif not numeric_data.empty:
        new_data = numeric_imputed
    elif not categorical_data.empty:
        new_data = categorical_imputed
    return new_data

# KNN imputer


def knn_impt(raw_data):
    from sklearn.impute import KNNImputer
    k_neighbors = st.slider("Choose the number of neighbors",
                            min_value=1, max_value=raw_data.shape[1])

    # Separate numeric and categorical columns
    numeric_data = raw_data.select_dtypes(include='number')
    categorical_data = raw_data.select_dtypes(include='object')

    if not categorical_data.empty:
        # encoder = OneHotEncoder(drop='first', sparse=False)
        categorical_encoded = pd.DataFrame(simple_imputer(
            categorical_data), columns=categorical_data.columns)

        # KNN Imputation for numeric columns
    if not numeric_data.empty:
        knn_imputer_numeric = KNNImputer(n_neighbors=k_neighbors)
        numeric_imputed = pd.DataFrame(knn_imputer_numeric.fit_transform(
            numeric_data), columns=numeric_data.columns)

    # Concatenate the imputed dataframes
    if not numeric_data.empty and not categorical_data.empty:
        new_data = pd.concat([numeric_imputed, categorical_encoded], axis=1)
    elif not numeric_data.empty:
        new_data = numeric_imputed
    elif not categorical_data.empty:
        new_data = categorical_encoded
    return new_data


def imputer(raw_data):
    st.title('Please Select Imputer')
    imputer = st.selectbox(
        'Imputer Type', ['None', 'Simple Imputer', 'KnnImputer'])
    if imputer == 'Simple Imputer':
        new_data = simple_imputer(raw_data)
        return new_data
    if imputer == 'KnnImputer':
        new_data = knn_impt(raw_data)
        return new_data
    if imputer == 'None':
        new_data = raw_data.copy()
        return new_data
    if st.button('Show Result'):
        st.dataframe(new_data)


# Selecet columns
def select_col(raw_data):
    st.title('Choose  Variable for building model')
    selected_variable = st.multiselect('Choose Variables', (raw_data.columns))
    raw_data1 = raw_data[selected_variable]
    return raw_data1
# One Hote Encoder


def encoder(raw_data1):
    st.title('Handling Categorical Data')
    from sklearn.preprocessing import LabelEncoder
    column1, column2 = st.columns(2)
    with column1:
        st.header('One Hot Encoding')
        cat_columns_onehot = st.multiselect('Choose Categorical Columns', raw_data1.select_dtypes(
            include='object').columns, key='onehot')
        if cat_columns_onehot:
            st.write("Onehot Encoded Columns", cat_columns_onehot)

    with column2:
        st.header('Label Encoder')
        cat_column_label = st.multiselect('Choose Categorical Columns', raw_data1.select_dtypes(
            include='object').columns, key='label')
        if cat_column_label:
            st.write("Seçilen Label Sütunu:", cat_column_label)
            cat_columns_onehot = [
                col for col in cat_columns_onehot if col != cat_column_label]

    label_encoded = pd.DataFrame()
    onehot_encoded = pd.DataFrame()
    if cat_columns_onehot:
        onehot_encoded = pd.get_dummies(
            raw_data1[cat_columns_onehot], drop_first=True)
        st.write("One Hot Encoded Veri:")
        # st.write(onehot_encoded)

    if cat_column_label:
        for i in range(len(cat_column_label)):
            label_encoder = LabelEncoder()
            label_encoded = pd.DataFrame(label_encoder.fit_transform(
                raw_data1[cat_column_label[i]]), columns=[cat_column_label[i]])
        st.write("Label Encoded Sütun:")
        # st.write(label_encoded)
    new_data1 = pd.concat([label_encoded, onehot_encoded, raw_data1.select_dtypes(exclude='object')],
                          axis=1)
    if st.button('Show Last Version'):
        st.dataframe(new_data1)
    return new_data1


def select_variables(new_data):
    X = None
    y = None
    selected_x_variables = st.multiselect(
        'Select X Variables', new_data.columns, key='selected_x_variables')
    y_variable_options = [
        col for col in new_data.columns if col not in selected_x_variables]
    y_variable = st.selectbox(
        'Select Y Variable', y_variable_options, key='y_variable')
    if selected_x_variables and y_variable:
        X = new_data[selected_x_variables]
        y = new_data[y_variable]
        if st.button('Show Indepented Variables'):
            st.dataframe(X)
        if st.button('Show Depented Variable'):
            st.dataframe(y)

    return (X, y)

# train_test_spilt


def train_test(X, y, cll_reg):
    from sklearn.model_selection import train_test_split
    test_size = st.slider('Choose Test Size', min_value=0.0,
                          max_value=1.0, step=0.01, value=0.2)
    random_state = st.number_input(
        'Random State', min_value=1, step=1, value=42, format='%d')
    if cll_reg == 'Classification':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state),
                                                            stratify=y, shuffle=True)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state),
                                                            shuffle=True)
    return (X_train, X_test, y_train, y_test)

# Scale


def scale(X_train, X_test):
    from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
    scale = st.radio('Select Scale Type', ('None',
                     'RobustScaler', 'MinMaxScaler', 'StandardScaler'))
    if scale == 'None':
        X_train_std = X_train
        X_test_std = X_test
    else:
        if scale == 'RobustScaler':
            scaler = RobustScaler()
        elif scale == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif scale == 'StandardScaler':
            scaler = StandardScaler()

        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)
    return (X_train_std, X_test_std)


def tree_models(X_train_std, X_test_std, y_train):
    from sklearn.tree import DecisionTreeClassifier
    max_depth = st.slider('Max Depth:', min_value=1,
                          max_value=10, value=3, step=1)
    criterion = st.radio('Criterion:', ['gini', 'entropy'])
    model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    model.fit(X_train_std, y_train)
    y_pred = model.predict(X_test_std)
    return (model, y_pred)


def svm_model(X_train_std, X_test_std, y_train):
    from sklearn.svm import SVC
    C = st.slider('C (Regularization Parameter):', min_value=0.1,
                  max_value=10.0, value=1.0, step=0.1)
    kernel = st.radio('Kernel:', ['linear', 'rbf', 'poly'])
    model = SVC(C=C, kernel=kernel)
    model.fit(X_train_std, y_train)
    y_pred = model.predict(X_test_std)
    return (model, y_pred)


def logistic_regression_model(X_train_std, X_test_std, y_train):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train_std, y_train)
    y_pred = model.predict(X_test_std)
    return (model, y_pred)


def naive_bayes_model(X_train_std, X_test_std, y_train):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X_train_std, y_train)
    y_pred = model.predict(X_test_std)
    return (model, y_pred)


def random_forest_model(X_train_std, X_test_std, y_train):
    from sklearn.ensemble import RandomForestClassifier
    n_estimators = st.slider('Number of Estimators:',
                             min_value=10, max_value=200, value=100, step=10)
    criterion = st.radio('Criterion:', ['gini', 'entropy'])
    model = RandomForestClassifier(
        n_estimators=n_estimators, criterion=criterion)
    model.fit(X_train_std, y_train)
    y_pred = model.predict(X_test_std)
    return (model, y_pred)


def k_nearest_neighbors_model(X_train_std, X_test_std, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    n_neighbors = st.slider('Number of Neighbors:',
                            min_value=1, max_value=20, value=5, step=1)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train_std, y_train)
    y_pred = model.predict(X_test_std)
    return (model, y_pred)


def classifer(X_train_std, X_test_std, y_train, y_test):
    # classification alghoritms
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    ch_model = st.radio('Choose Classification Models', ('None', 'Super Vector Machine(SVM)',
                        'Decision Tree', 'Logistic Regression', 'Naive Bayes', 'Random Forest', 'K-Nearest Neighbors'))
    # Decsion tree
    if ch_model == 'Decision Tree':
        tree = tree_models(X_train_std, X_test_std, y_train)
        report = classification_report(y_test, tree[1], output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report)
    elif ch_model == 'Super Vector Machine(SVM)':
        svm = svm_model(X_train_std, X_test_std, y_train)
        report = classification_report(y_test, svm[1], output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report)
    elif ch_model == 'Logistic Regression':
        logistic_result = logistic_regression_model(
            X_train_std, X_test_std, y_train)
        logistic_report = classification_report(
            y_test, logistic_result[1], output_dict=True)
        df_logistic_report = pd.DataFrame(logistic_report).transpose()
        st.dataframe(df_logistic_report)
    elif ch_model == 'Naive Bayes':
        naive_bayes_result = naive_bayes_model(
            X_train_std, X_test_std, y_train)
        naive_bayes_report = classification_report(
            y_test, naive_bayes_result[1], output_dict=True)
        df_naive_bayes_report = pd.DataFrame(naive_bayes_report).transpose()
        st.dataframe(df_naive_bayes_report)
    elif ch_model == 'Random Forest':
        random_forest_result = random_forest_model(
            X_train_std, X_test_std, y_train)
        random_forest_report = classification_report(
            y_test, random_forest_result[1], output_dict=True)
        df_random_forest_report = pd.DataFrame(
            random_forest_report).transpose()
        st.dataframe(df_random_forest_report)
    elif ch_model == 'K-Nearest Neighbors':
        knn_result = k_nearest_neighbors_model(
            X_train_std, X_test_std, y_train)
        knn_report = classification_report(
            y_test, knn_result[1], output_dict=True)
        df_knn_report = pd.DataFrame(knn_report).transpose()
        st.dataframe(df_knn_report)
    elif ch_model == 'Other':
        st.header('The project is in the process of development')

# Build Model


def build_model(raw_data):
    st.markdown(
        """
    <div style='background-color: #4B0082; padding: 10px; border-radius: 10px;'>
        <h2 style='color: white; font-weight: bold;'>Build Model</h2>
        <p style='color: white;'>Here, you can explore and build your machine learning model.
        Consider providing input features, selecting algorithms, and training your model.</p>
    </div>
    """,
        unsafe_allow_html=True
    )

    raw_data1 = select_col(raw_data)
    if not raw_data1.empty:
        new_data = imputer(raw_data1)
        st.dataframe(new_data)
        new_data = encoder(new_data)
        x_y = select_variables(new_data)
        if x_y[0] is not None and not x_y[0].empty:
            # Classification function begining
            cll_reg = st.radio('Choose Classification or Regression',
                               ('None', 'Classification', 'Regression'))
            if cll_reg == 'Classification':
                is_integer = np.all(x_y[1].astype(int) == x_y[1])
                unique_values, counts = np.unique(x_y[1], return_counts=True)
                if is_integer and len(unique_values) < 20:
                    xt_yt = train_test(x_y[0], x_y[1], cll_reg)
                    X_train, X_test, y_train, y_test = xt_yt[0], xt_yt[1], xt_yt[2], xt_yt[3]
                    scaled_data = scale(X_train, X_test)
                    X_train_std, X_test_std = scaled_data[0], scaled_data[1]
                    classifer(X_train_std, X_test_std, y_train, y_test)
                else:
                    st.title("The target values you selected contain continuous values, resulting in the creation of too many classes. Please make sure you have chosen the correct target variable.")
            elif cll_reg == 'Regression':
                st.title('The project is in the process of development')
            else:
                pass


def data_analysis(raw_data):
    analyzing = st.sidebar.selectbox(
        'Start Analyzing', ('None', 'Show Data', 'EDA', 'Build Model'))
    if 'Show Data' == analyzing:
        show_data(raw_data)
    if 'EDA' == analyzing:
        eda(raw_data)
    if 'Build Model' == analyzing:
        build_model(raw_data)


# =============================================================================
# ### Begining
# =============================================================================
if page == 'Load Dataset':
    upload_data = st.file_uploader("Load Your Data", type=["csv"])
    if upload_data is not None:
        raw_data = pd.read_csv(upload_data)
        analyzing = st.sidebar.selectbox(
            'Start Analyzing', ('None', 'Show Data', 'EDA', 'Build Model'))
        if 'Show Data' == analyzing:
            show_data(raw_data)
        if 'EDA' == analyzing:
            eda(raw_data)
        if 'Build Model' == analyzing:
            build_model(raw_data)
elif page == 'Explore Ready Dataset':
    datasets = st.radio('Choose Dataset', ('Iris', 'Breast Cancer', 'Wine'))

    if datasets == 'Iris':
        from sklearn.datasets import load_iris
        data = load_iris()
    elif datasets == 'Boston':
        from sklearn.datasets import load_boston
        data = load_boston()
    elif datasets == 'Breast Cancer':
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
    elif datasets == 'Wine':
        from sklearn.datasets import load_wine
        data = load_wine()
    raw_data = pd.DataFrame(data=data.data, columns=data.feature_names)

    if hasattr(data, 'target'):
        raw_data['target'] = data.target

    data_analysis(raw_data)

else:
    st.title('The project is in the process of development')
