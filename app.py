# import test
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import calendar
from sklearn.preprocessing import LabelEncoder
import io
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from boruta import BorutaPy
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Decision Support System Project')
st.title('Laundry Scenario')
st.text('Lim Zhi Xuan 1171102256')
st.text('Chai Kar Sing 1171101353')
st.text('Gan Xin Yun 1171101072')


st.sidebar.header('Section')
section = st.sidebar.radio("Choose a section:", 
                              ("Exploratory Data Analysis", "Feature Selection", "Classification")
                              )


df = pd.read_csv("LAUNDRY.csv")

if section =="Exploratory Data Analysis":

    # show raw data
    st.subheader('Raw data')
    st.write(df)

    dftemp = df.copy()
    st.subheader("Select specific row that want to view")
    select_row = st.number_input('Select row: ', min_value=0,
                           max_value=len(dftemp), value=0, key=0)
    dftemp = dftemp.iloc[select_row]
    st.write(dftemp.astype('object'))

    # check N/A
    st.subheader('For Checking N/A')
    st.write(df.isna().sum(axis=0))

    st.subheader("Add External Data")
    df['Money_Spent'] = np.random.randint(1, 50, df.shape[0])
    st.write(df.head(10))


    st.header('Data Pre-Processing')
    st.subheader('Fill N/A with Unknown')
    # fill with unknown
    df1 = df.replace(np.nan, 'Unknown', regex=True)
    st.write(df1)  # might change to KNN imputer to show numerical data
    # df1.drop([ ], 1)

    # Number of nan after fill in unknown
    st.subheader('The number of Nan value after replace with "Unknown"')
    st.write(df1.isna().sum(axis=0))

    # Perform label encoding
    st.subheader('Label Encoding')
    df1 = df1.drop(['No', 'Washer_No', 'Dryer_No', 'Age_Range'], 1)
    col = df1.columns

    st.header("Data visuallization")
    le = preprocessing.LabelEncoder()
    df1[col] = df1[col].apply(lambda col: le.fit_transform(col))
    df1 = pd.concat([df[['Age_Range', 'Washer_No', 'Dryer_No']], df1], 1)

    imputer = KNNImputer(n_neighbors=5)
    df1['Age_Range'] = imputer.fit_transform(df1)
    df1['Age_Range'] = df1['Age_Range'].round(0)
    st.write(df1)

    # add external data money spent
    df1['Money_Spent'] = np.random.randint(1, 50, df1.shape[0])

    # Data Visualisation


    # Show wash item
    sns.set_theme(style="darkgrid")
    ax = sns.countplot(x="Wash_Item", data=df1)
    st.pyplot()

    # Show data type (OPTIONAL)
    # st.subheader('The data types of each features')
    # df1.dtypes


    # Show Item washed in different washer
    st.subheader('Item washed in diffrent washer ')
    a = pd.crosstab(df1.Washer_No, df1.Wash_Item).plot(kind='bar', stacked=True)
    # plt.bar(color="#0000FF", edgecolor='white', width=1, label = 'Blanket')
    plt.title('Item washed in different washer')
    plt.xlabel('Washer No')
    plt.ylabel('Number of Wash Item')
    st.pyplot()

    # Show Item washer in different dryer
    st.subheader('Item washed in diffrent dryer ')
    a = pd.crosstab(df1.Dryer_No, df1.Wash_Item).plot(kind='bar', stacked=True)
    plt.title('Item washed in different dryer')
    plt.xlabel('Dryer No')
    plt.ylabel('Number of Wash Item')
    st.pyplot()

    # Correlation graph
    st.subheader("Simple Correlation Plot with Seaborn ")
    plt.figure(figsize=(13, 13))
    st.write(sns.heatmap(df1.corr(), vmax=.8, square=True, annot=True,
                        fmt='.2f', annot_kws={'size': 12}, cmap=sns.color_palette("Blues")))

    # Use Matplotlib to render seaborn
    st.pyplot()


    # Handling imbalanced data with SMOTE
    X1 = df1.drop('Wash_Item', 1)
    y1 = df1.Wash_Item

    features = X1.columns
    os = SMOTE(sampling_strategy="not majority", k_neighbors=5, random_state=10)
    X_train_os, X_test_os, y_train_os, y_test_os = train_test_split(
        X1, y1.values.ravel(), test_size=0.2, random_state=10)
    X1, y1 = os.fit_resample(X_train_os, y_train_os)
    X1 = pd.DataFrame(data=X1, columns=features)
    y1 = pd.DataFrame(data=y1, columns=['Wash_Item'])
    y1["Wash_Item"].value_counts().plot(kind="bar")
    plt.title("Wash_Item")
    st.write(y1["Wash_Item"].value_counts())
    y1 = pd.Series(y1['Wash_Item'].values)

    st.subheader("Wash_Item data with SMOTE")
    st.pyplot()

    

    
#=========================================================================================

elif section =="Feature Selection":
    df1 = df.replace(np.nan, 'Unknown', regex=True)
    df1 = df1.drop(['No', 'Washer_No', 'Dryer_No', 'Age_Range'], 1)

    col = df1.columns

    le = preprocessing.LabelEncoder()
    df1[col] = df1[col].apply(lambda col: le.fit_transform(col))
    df1 = pd.concat([df[['Age_Range', 'Washer_No', 'Dryer_No']], df1], 1)


    imputer = KNNImputer(n_neighbors=5)
    df1['Age_Range'] = imputer.fit_transform(df1)
    df1['Age_Range'] = df1['Age_Range'].round(0)
    df1['Money_Spent'] = np.random.randint(1, 50, df1.shape[0])

    X = df1.drop('Wash_Item', 1)
    y = df1.Wash_Item

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10)

    # For data wihout over-sampled
    st.subheader('Feature Selection')



    def ranking(ranks, names, order=1):
        minmax = MinMaxScaler()
        ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
        ranks = map(lambda x: round(x, 2), ranks)
        return dict(zip(names, ranks))

    X1 = df1.drop('Wash_Item', 1)
    y1 = df1.Wash_Item
    colnames = X.columns
    

    rf = RandomForestClassifier(
        n_jobs=-1, class_weight="balanced", random_state=100, max_depth=3)

    feat_selector = BorutaPy(rf, n_estimators="auto", random_state=100)

    feat_selector.fit(X1.values, y1.values.ravel())

    # Getting the score
    boruta_score = ranking(
        list(map(float, feat_selector.ranking_)), colnames, order=-1)
    boruta_score = pd.DataFrame(
        list(boruta_score.items()), columns=['Features', 'Score'])
    boruta_score = boruta_score.sort_values("Score", ascending=False)

    st.text("Top 10 features from datasets")
    #        Show top 10
    st.write(boruta_score.head(10))     

    #       show bottom 10
    st.text("Bottom 10 features from datasets")
    st.write(boruta_score.tail(10))

    #       Boruta graph
    sns_boruta_plot = sns.catplot(x="Score", y="Features", data=boruta_score[0:15], kind="bar",
                                height=14, aspect=1.9, palette='coolwarm')
    st.subheader("Top 20 Boruta Features")
    plt.title("Boruta Features")
    st.pyplot()

#====================================================================================

elif section== "Classification":
    df1 = df.replace(np.nan, 'Unknown', regex=True)
    df1 = df1.drop(['No', 'Washer_No', 'Dryer_No', 'Age_Range'], 1)

    col = df1.columns

    le = preprocessing.LabelEncoder()
    df1[col] = df1[col].apply(lambda col: le.fit_transform(col))
    df1 = pd.concat([df[['Age_Range', 'Washer_No', 'Dryer_No']], df1], 1)


    imputer = KNNImputer(n_neighbors=5)
    df1['Age_Range'] = imputer.fit_transform(df1)
    df1['Age_Range'] = df1['Age_Range'].round(0)
    df1['Money_Spent'] = np.random.randint(1, 50, df1.shape[0])

    X = df1.drop('Wash_Item', 1)
    y = df1.Wash_Item

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10)

    # For data wihout over-sampled



    def ranking(ranks, names, order=1):
        minmax = MinMaxScaler()
        ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
        ranks = map(lambda x: round(x, 2), ranks)
        return dict(zip(names, ranks))


    colnames = X.columns

    rf = RandomForestClassifier(
        n_jobs=-1, class_weight="balanced", random_state=100, max_depth=3)

    feat_selector = BorutaPy(rf, n_estimators="auto", random_state=100)

    feat_selector.fit(X.values, y.values.ravel())

    # Getting the score
    boruta_score = ranking(
        list(map(float, feat_selector.ranking_)), colnames, order=-1)
    boruta_score = pd.DataFrame(
        list(boruta_score.items()), columns=['Features', 'Score'])
    boruta_score = boruta_score.sort_values("Score", ascending=False)

    X1 = df1.drop('Wash_Item', 1)
    y1 = df1.Wash_Item

    features = X1.columns
    os = SMOTE(sampling_strategy="not majority", k_neighbors=5, random_state=10)
    X_train_os, X_test_os, y_train_os, y_test_os = train_test_split(
        X1, y1.values.ravel(), test_size=0.2, random_state=10)
    X1, y1 = os.fit_resample(X_train_os, y_train_os)
    X1 = pd.DataFrame(data=X1, columns=features)
    y1 = pd.DataFrame(data=y1, columns=['Wash_Item'])
    y1["Wash_Item"].value_counts().plot(kind="bar")
    y1 = pd.Series(y1['Wash_Item'].values)


    st.subheader('Classification with different technique')
    #       NAIVE BAISE
    nb = GaussianNB()
    nb.fit(X, y)
    st.write('NAIVE BAISE')

    status = st.radio("Select condition: ",
                    ('Before SMOTE data', 'After SMOTE data'))
    if (status == 'After SMOTE data'):
        nb = GaussianNB()
        nb.fit(X1, y1)
        st.write('NB Score (train)=', nb.score(X_train_os, y_train_os))
        st.write('NB Score (test)=', nb.score(X_test_os, y_test_os))
    else:
        st.write('NB Score (train)=', nb.score(X_train, y_train))
        st.write('NB Score (test)=', nb.score(X_test, y_test))


    prob_NB = nb.predict_proba(X_test)
    prob_NB = prob_NB[:, 1]

    prob_NB_os = nb.predict_proba(X_test_os)
    prob_NB_os = prob_NB_os[:, 1]

    fpr_NB, tpr_NB, thresholds_NB = roc_curve(y_test, prob_NB, pos_label=2)
    fpr_NB_os, tpr_NB_os, thresholds_NB_os = roc_curve(
        y_test_os, prob_NB_os, pos_label=2)

    plt.plot(fpr_NB, tpr_NB, color='orange', label='NB')
    plt.plot(fpr_NB_os, tpr_NB_os, color='blue', label='NB(SMOTE)')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    st.pyplot(plt)


    #       RANDOM FOREST CLASSIFIER
    st.write('RANDOM FOREST CLASSIFIER')
    rf = RandomForestClassifier(n_estimators=5, random_state=10)
    rf.fit(X1, y1)

    status = st.radio("Select condition: ",
                    ('Before SMOTE data ', 'After SMOTE data '))
    if (status == 'After SMOTE data '):
        y_pred = rf.predict(X_test_os)
        st.write('RF Score (train)=', rf.score(X_train_os, y_train_os))
        st.write('RF Score (test)=', rf.score(X_test_os, y_test_os))
    else:
        y_pred = rf.predict(X_test)
        st.write('RF Score (train)=', rf.score(X_train, y_train))
        st.write('RF Score (test)=', rf.score(X_test, y_test))


    #       SVM
    st.write('SUPPORT VECTOR MACHINES')
    model_svm = svm.SVC(kernel='rbf', gamma='auto', degree=2/3)
    model_svm.fit(X, y)

    status = st.radio("Select condition: ",
                    ('Before SMOTE data  ', 'After SMOTE data  '))
    if (status == 'After SMOTE data  '):
        y_pred3 = model_svm.predict(X_test_os)
        y_pred4 = model_svm.predict(X_train_os)
        st.write("SVM Score (train)=", metrics.accuracy_score(y_train_os, y_pred4))
        st.write("SVM Score (test)=", metrics.accuracy_score(y_test_os, y_pred3))
    else:
        st.write('The accurary of SUPPORT VECTOR MACHINES')
        y_pred = model_svm.predict(X_test)
        y_pred2 = model_svm.predict(X_train)
        st.write("SVM Score (train)=", metrics.accuracy_score(y_train, y_pred2))
        st.write("SVM Score (test)=", metrics.accuracy_score(y_test, y_pred))


    #       KNN
    knn = KNeighborsClassifier(n_neighbors=10)

    status = st.radio("Select condition: ",
                    ('Before SMOTE data   ', 'After SMOTE data   '))
    if (status == 'After SMOTE data   '):
        knn.fit(X_train_os, y_train_os)
        st.write("Accuracy: ", knn.score(X_train_os, y_train_os))
        st.write("Accuracy: ", knn.score(X_test_os, y_test_os))
    else:
        knn.fit(X_train, y_train)
        st.write("Accuracy: ", knn.score(X_train, y_train))
        st.write("Accuracy: ", knn.score(X_test, y_test))


    # Classification after dropping low score features
    st.subheader("Classification after dropping low score features")
    X2 = df1.drop(['Spectacles', 'Wash_Item', 'Time', 'Gender', 'pants_type',
                'shirt_type', 'Washer_No', 'Basket_Size', 'Race', 'Attire'], 1)
    y2 = df1.Wash_Item

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X2, y2, test_size=0.2, random_state=10)

    X = df1.drop('Wash_Item', 1)
    y = df1.Wash_Item

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10)

    status = st.radio("Select Classifier: ", ('NAIVE BAYES',
                                            'RANDOM FOREST CLASSIFIER', 'K-NEAREST NEIGHBORS ALGORITHM', 'SUPPORT VECTOR MACHINES'))

    if (status == 'NAIVE BAYES'):
        #      NB
        st.write("NAIVE BAYES")
        nb = GaussianNB()
        nb.fit(X2, y2)

        st.write('NB Score (train)=', nb.score(X_train2, y_train2))
        st.write('NB Score (test)=', nb.score(X_test2, y_test2))

    elif (status == 'RANDOM FOREST CLASSIFIER'):
        #       RF
        st.write('RANDOM FOREST CLASSIFIER')
        rf = RandomForestClassifier(n_estimators=3, random_state=10)
        rf.fit(X2, y2)

        y_pred = rf.predict(X_test2)

        st.write('RF Score (train)=', rf.score(X_train2, y_train2))
        st.write('RF Score (test)=', rf.score(X_test2, y_test2))

    elif (status == 'K-NEAREST NEIGHBORS ALGORITHM'):
        #       KNN
        st.write('K-NEAREST NEIGHBORS ALGORITHM')
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(X2, y2)
        st.write("Accuracy (train)= ", knn.score(X_train2, y_train2))
        st.write("Accuracy (test)= ", knn.score(X_test2, y_test2))

    elif (status == 'SUPPORT VECTOR MACHINES'):
        #       SVM
        st.write('SUPPORT VECTOR MACHINES')
        model_svm = svm.SVC(kernel='rbf', gamma='auto', degree=2/3)
        model_svm.fit(X2, y2)

        y_pred3 = model_svm.predict(X_test2)
        y_pred23 = model_svm.predict(X_train2)

        st.write("SVM Score (train)= ", metrics.accuracy_score(y_train2, y_pred23))
        st.write("SVM Score (test)= ", metrics.accuracy_score(y_test2, y_pred3))

    #       PREDICT
    st.subheader("Predict using SUPPORT VECTOR MACHINES")


    #       CLUSTERING
    sns.relplot(x="Age_Range", y="Money_Spent", hue="Wash_Item", sizes=(
        40, 400), alpha=0.5, palette="muted", height=6, data=df1)
    plt.title('Clustering')
    st.pyplot()

    #       K-MEANS

    km = KMeans(n_clusters=2, random_state=1)
    km.fit(X)

    distortions = []

    for i in range(1, 11):
        km = KMeans(
            n_clusters=i, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        km.fit(X)
        distortions.append(km.inertia_)

    # plot
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    st.pyplot()

    # another graph
    df_new = df1.copy()
    df_new = df_new.drop("Wash_Item", axis=1)
    df_new['Wash_Item'] = km.labels_


    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    st.subheader("Scatter plot of wash item by age range and money spent")
    sns.relplot(x="Age_Range", y="Money_Spent", hue="Wash_Item",
                data=df1, ax=axes[0], sizes=(40, 400), alpha=0.5)
    plt.title('Cluster 1')
    st.pyplot()

    sns.relplot(x="Age_Range", y="Money_Spent", hue="Wash_Item",
                data=df_new, ax=axes[1], sizes=(40, 400), alpha=0.5)
    plt.title('Cluster 2')
    st.pyplot()
