import streamlit as st
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn.datasets
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Processing the Raw Data
# numerical_columns_list = []
# categorical_columns_list = []
# for i in dataset.columns:
#     if dataset[i].dtype == np.dtype("float64") or dataset[i].dtype == np.dtype("int64"):
#         numerical_columns_list.append(dataset[i])
#     else:
#         categorical_columns_list.append(dataset[i])

# numerical_data = pd.concat(numerical_columns_list, axis=1)
# categorical_data = pd.concat(categorical_columns_list, axis=1)

def add_parameter_ui(class_name):
    params = dict()
    if class_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif class_name == 'Logistic Regression':
        pass
    # elif class_name == "SVM":
    #     C = st.sidebar.slider("C", 0.01, 10.0)
    #     params["C"] = C
    # elif class_name == 'Random Forest':
    #     max_depth = st.sidebar.slider("Max_Depth", 2, 100)
    #     n_estimators = st.sidebar.slider("N-Estimators", 1, 1000)
    #     params["max_depth"] = max_depth
    #     params["n_estimators"] = n_estimators
    # elif class_name == 'Decision Tree':
    #     max_depth = st.sidebar.slider("Max_Depth", 2, 100)
    #     params["max_depth"] = max_depth
    return params


def get_classifier(class_name, params):
    if class_name == "KNN":
        classifier = KNeighborsClassifier(n_neighbors=params["K"])
    elif class_name == "Logistic Regression":
        classifier = LogisticRegression()
    # elif class_name == "SVM":
    #     classifier = SVC(C=params["C"])
    # elif class_name=="Random Forest":
    #     classifier = RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"], random_state=0)
    # elif class_name=="Decision Tree":
    #     classifier = DecisionTreeClassifier(max_depth=params["max_depth"])
    return classifier


accuracies = []

# Display a title
st.title('Principal Components Analysis Visualization using Streamlit')

# List all the sklearn datasets
dataset_list = ['Iris', 'Breast Cancer', 'Wine Quality']
dataset_dict = {
    'Iris': sklearn.datasets.load_iris(as_frame=True),
    'Breast Cancer': sklearn.datasets.load_breast_cancer(as_frame=True),
    'Wine Quality': sklearn.datasets.load_wine(as_frame=True)
}

# Choose Dataset for PCA
dataset_selection = st.sidebar.selectbox(
    "Select Dataset from the List ", dataset_list, 0)

# classifier_name = st.sidebar.selectbox(
#     "Select the classifier", ("KNN", "SVM", "Random Forest", "Decision Tree", "Logistic Regression"))
classifier_name = st.sidebar.selectbox(
    "Select the classifier", ("KNN", "Logistic Regression"))
params = add_parameter_ui(classifier_name)

# Display Raw Data
st.subheader('Raw data')
dataset = dataset_dict[dataset_selection]
df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
df['target'] = pd.Categorical.from_codes(dataset.target, dataset.target_names)
st.write(df)

X = dataset.data
y = dataset.target

st.subheader('Explore the original data')
st.write('Shape of Dataset', X.shape)
st.write('Number of classes', len(np.unique(y)))

xvar = st.selectbox('Select x-axis:', dataset.feature_names)
yvar = st.selectbox('Select y-axis:', dataset.feature_names)
colors = st.selectbox('Select colors:', dataset.target_names)

st.write(px.scatter(df, x=xvar, y=yvar, color='target'))

st.subheader("Correaltion Matrix of the data")
corr = df.corr()

fig_size_val = 4
if(dataset_selection == 'Iris'):
    fig_size_val = 4
elif(dataset_selection == 'Breast Cancer'):
    fig_size_val = 21
elif(dataset_selection == 'Wine Quality'):
    fig_size_val = 15

fig = plt.figure(figsize=(fig_size_val, fig_size_val))
sns.heatmap(corr, xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True)
st.pyplot(fig)

# Classification Before PCA
st.title("Classification Before PCA using {}".format(classifier_name))
classifier = get_classifier(classifier_name, params)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

st.write(f"Classifier = {classifier_name}")
conf_mat = confusion_matrix(y_test, y_pred)
st.write('Confusion matrix: ', conf_mat)
st.write('Accuracy  =  ', accuracy_score(y_test, y_pred))
# st.write('Precision =  ', precision_score(y_test, y_pred))
# st.write('Recall    =  ', recall_score(y_test, y_pred))
ac1 = accuracy_score(y_test, y_pred)

# Display a title
st.title(f'PCA applied to {dataset_selection}')


# Number of principal components
st.sidebar.markdown(
    r"""
    Note: The number is nonnegative integer.
    """
)

num_pca = int(st.sidebar.number_input(
    'The minimum value is an integer of 3 or more.', value=3, step=1, min_value=3))

# Scaling the data
df_scaled = StandardScaler().fit_transform(df.drop('target', axis=1))

# Perform PCA
pca = PCA(n_components=num_pca)
df_transformed = pca.fit_transform(df_scaled)

col_names = [i+1 for i in range(df_transformed.shape[1])]

df_transformed_df = pd.DataFrame(df_transformed, columns=col_names)
df_transformed_df = pd.concat([df_transformed_df, df['target']], axis=1)

st.subheader('Transformed data')
st.write(df_transformed_df)

# Explore PCA Components
st.subheader('Explore principal components')

xvar = st.sidebar.selectbox("Select X-axis: ", np.arange(1, num_pca+1), 0)
yvar = st.sidebar.selectbox("Select Y-axis: ", np.arange(1, num_pca+1), 1)
zvar = st.sidebar.selectbox("Select Z-axis: ", np.arange(1, num_pca+1), 2)

st.header(f'2D Representation - PCA {xvar} vs PCA {yvar}')
st.write(px.scatter(df_transformed_df, x=xvar, y=yvar, color='target'))

st.header(f'3D Representation - PCA {xvar} vs PCA {yvar} vs PCA {zvar}')
x_lbl, y_lbl, z_lbl = f"{xvar}", f"{yvar}", f"{zvar}"
x_plot, y_plot, z_plot = df_transformed[:, xvar -
                                        1], df_transformed[:, yvar-1], df_transformed[:, zvar-1]

# Create an object for 3d scatter
trace1 = go.Scatter3d(
    x=x_plot, y=y_plot, z=z_plot,
    mode='markers',
    marker=dict(
        size=5,
        color=y,
        colorscale='Viridis'
    )
)
# Create an object for graph layout
fig = go.Figure(data=[trace1])
fig.update_layout(scene=dict(
    xaxis_title=x_lbl,
    yaxis_title=y_lbl,
    zaxis_title=z_lbl),
    width=700,
    margin=dict(r=20, b=10, l=10, t=10),
)
# Plot on the dashboard on streamlit
st.plotly_chart(fig, use_container_width=True)


# Explore Loadings
# st.subheader('Explore loadings')
# loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
# loadings_df = pd.DataFrame(loadings, columns=col_names)
# loadings_df = pd.concat(
#     [loadings_df, pd.Series(dataset.feature_names, name='var')], axis=1)
# st.write(loadings_df)

# component = st.selectbox(
#     'Select component:', loadings_df.columns[:-1])

# bar_chart = px.bar(loadings_df[['var', component]].sort_values(component),
#                    y='var',
#                    x=component,
#                    orientation='h',
#                    range_x=[-1, 1])


# st.write(bar_chart)

# Classification After PCA
st.title("Classification After PCA using {}".format(classifier_name))
classifier = get_classifier(classifier_name, params)

X = df_transformed_df.drop('target', axis=1)
y = df_transformed_df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

st.write(f"Classifier = {classifier_name}")
conf_mat = confusion_matrix(y_test, y_pred)
st.write('Confusion matrix: ', conf_mat)
st.write('Accuracy  =  ', accuracy_score(y_test, y_pred))
# st.write('Precision =  ', precision_score(y_test, y_pred))
# st.write('Recall    =  ', recall_score(y_test, y_pred))
ac2 = accuracy_score(y_test, y_pred)


st.subheader('Comparison between classifiers - ')
st.write(f"Classifier = {classifier_name}")
st.write(f"Classifier Details = {classifier}")
st.write("Dataset = {}".format(dataset_selection))
st.write("Dataset Shape Before PCA : ", dataset.data.shape)
st.write("Dataset Shape After PCA : ", df_transformed.shape)

# Plot accuracy in streamlit bar chart
accuracy_list = [['Before PCA', ac1], ['After PCA', ac2]]
accuracies_df = pd.DataFrame(accuracy_list, columns=['PCA', 'Accuracy'])
st.write(px.bar(accuracies_df, x='PCA', y='Accuracy',
                title="Accuracy Comparison - Before and After PCA", height=800))
