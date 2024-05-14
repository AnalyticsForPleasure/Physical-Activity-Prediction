# https://www.kaggle.com/datasets/diegosilvadefrana/fisical-activity-dataset

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import logging


####### The column names are:
# 'activityID', 'heart_rate', 'hand temperature (°C)', 'hand acceleration X ±16g', 'hand acceleration Y ±16g', 'hand acceleration Z ±16g', 'hand gyroscope X', 'hand gyroscope Y', 'hand gyroscope Z', 'hand magnetometer X', 'hand magnetometer Y', 'hand magnetometer Z',
# 'chest temperature (°C)', 'chest acceleration X ±16g', 'chest acceleration Y ±16g', 'chest acceleration Z ±16g', 'chest gyroscope X', 'chest gyroscope Y', 'chest gyroscope Z', 'chest magnetometer X', 'chest magnetometer Y', 'chest magnetometer Z',
# 'ankle temperature (°C)', 'ankle acceleration X ±16g', 'ankle acceleration Y ±16g', 'ankle acceleration Z ±16g', 'ankle gyroscope X', 'ankle gyroscope Y', 'ankle gyroscope Z', 'ankle magnetometer X', 'ankle magnetometer Y', 'ankle magnetometer Z',
#  'PeopleId']

if __name__ == '__main__':
    pd.set_option('display.max_rows', 5000)
    pd.set_option('display.max_columns', 50)

    df = pd.read_csv('/home/shay/PycharmProjects/physical_activity_prediction/dataset/dataset2.csv')
    print(f'columns are: {df.columns}')

    res = pd.unique(df['activityID'])

    column_headers = df.columns.values.tolist()

    # for column in df.columns:
    #     print(f'The {column} consists of: {df[column].value_counts()}')

    print(df.describe(include='all'))

    # See how many missing values appear in the dataframe
    print(df.isnull().sum())

    print(f'The number of rows is: {df.shape[0]:,}')
    df = df.dropna()
    print(f'The number of rows is: {df.shape[0]:,} after dropping NAs')
    print('*')

    le = preprocessing.LabelEncoder()
    df['activityID'] = le.fit_transform(df['activityID'])

    y = df['activityID']
    X = df.loc[:, df.columns.drop('activityID')]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print()
    model = DecisionTreeClassifier()
    model.fit(X=X_train, y=y_train)
    y_predicted = model.predict(X=X_test)
    cm = confusion_matrix(y_test, y_predicted)
    labels = [label.replace(' ', '\n') for label in list(le.classes_)]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(values_format=',', cmap='GnBu')
    print(f'The accuracy is: {accuracy_score(y_true=y_test, y_pred=y_predicted):.3f}')
    print(list(le.inverse_transform(list(range(13)))))
    plt.title('Confusion Matrix')
    plt.xticks(rotation=20)
    plt.yticks(rotation=20)
    plt.show()
