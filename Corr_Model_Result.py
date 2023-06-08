import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def predict_model_survey(model):
    # Average Price of different Airlnes from Source city to Destination city
    df.groupby(['airline', 'source_city', 'destination_city'], as_index=False)['price'].mean().head(10)
    # Creating a Back up File
    df_bk = df.copy()
    # Coverting the labels into a numeric form using Label Encoder
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    #     # storing the Dependent Variables in X and Independent Variable in Y
    x = df.drop(['price'], axis=1)
    # x = df[['airline','flight','source_city','departure_time','arrival_time']['destination_city']['duration']
    y = df['price']
    # Splitting the Data into Training set and Testing Set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=42)
    # Scaling the values to convert the int values to Machine Languages
    from sklearn.preprocessing import MinMaxScaler
    mmscaler = MinMaxScaler(feature_range=(0, 1))
    x_train = mmscaler.fit_transform(x_train)
    x_test = mmscaler.fit_transform(x_test)
    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)

    model_train = model
    model_train.fit(x_train, y_train)
    # Predict the model with test data
    y_pred = model_train.predict(x_test)
    print(len(y_pred))
    return y_pred
def actual_value():
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    #     # storing the Dependent Variables in X and Independent Variable in Y
    x = df.drop(['price'], axis=1)
    y = df['price']
    # Splitting the Data into Training set and Testing Set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=42)
    print(len(y_test))
    return y_test
def plot_1_algorithm_corr(model, y_pred):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    import matplotlib.pyplot as plt
    # Vẽ biểu đồ
    y_test = actual_value()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(y_test, y_pred, c='r', marker='o', label=f'{model}')
    ax.scatter(y_test, y_test, c='b', marker='*', label=f"Actual Value")
    ax.set_xlabel('Actual value')
    ax.set_ylabel('Predicted value')
    plt.legend(loc='upper left')
    plt.show()
def plot_2_algorithms_corr(model_1, model_2, y_pred_1, y_pred_2):
    y_test = actual_value()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # Vẽ biểu đồ
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(y_test, y_pred_1, c='r', marker='o', label=f'{model_1}')
    ax.scatter(y_test, y_pred_2, c='b', marker='^', label=f'{model_2}')
    ax.scatter(y_test, y_test, c='g', marker='*', label=f'Actual value')
    ax.set_xlabel('Actual value')
    ax.set_ylabel(f'{model_1}')
    ax.set_zlabel(f'{model_2}')
    plt.legend(loc='upper left')
    plt.show()

def plot_3_algorithms_corr(model_1, model_2,model_3, y_pred_1, y_pred_2, y_pred_3):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    #     # storing the Dependent Variables in X and Independent Variable in Y
    x = df.drop(['price'], axis=1)
    y = df['price']
    # Splitting the Data into Training set and Testing Set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=42)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # Vẽ biểu đồ
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(y_test, y_pred_1, y_pred_2, c='r', marker='o', label=f'{model_1}')
    ax.scatter(y_test, y_pred_2, y_pred_3, c='b', marker='^', label=f'{model_2}')
    ax.scatter(y_test, y_pred_3, y_pred_1, c='g', marker='s', label=f'{model_3}')
    ax.set_xlabel('Actual value')
    ax.set_ylabel(f'{model_1}')
    ax.set_zlabel(f'{model_2, model_3}')
    plt.legend(loc='upper left')
    plt.show()

def plot_4_algorithms_corr(model_1, model_2,model_3,model_4, y_pred_1, y_pred_2, y_pred_3, y_pred_4):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    #     # storing the Dependent Variables in X and Independent Variable in Y
    x = df.drop(['price'], axis=1)
    y = df['price']
    # Splitting the Data into Training set and Testing Set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=42)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(y_test, y_pred_1, y_pred_2, c='r', marker='o', label=f'{model_1}')
    ax.scatter(y_test, y_pred_2, y_pred_3, c='b', marker='^', label=f'{model_2}')
    ax.scatter(y_test, y_pred_3, y_pred_4, c='g', marker='s', label=f'{model_3}')
    ax.scatter(y_test, y_pred_4, y_pred_1, c='y', marker='*', label=f'{model_4}')

    ax.set_xlabel('Actual value')
    ax.set_ylabel(f'{model_1, model_2}')
    ax.set_zlabel(f'{model_3, model_4}')
    plt.legend(loc='upper left')
    plt.show()
if __name__ == "__main__":
    df = pd.read_csv('Dataset/Clean_Dataset.csv')
    df = df.drop('Unnamed: 0', axis=1)
    y1 = predict_model_survey(LinearRegression())
    y2 = predict_model_survey(DecisionTreeRegressor())
    y3 = predict_model_survey(RandomForestRegressor())
    y4 = predict_model_survey(KNeighborsRegressor(n_neighbors=5))
    # plot_algorithms_corr(y1, y2)
    # Predict_Flight_Fare('1','0','6E-2046',  '1', '2', '1', '1', '2', '1', '2.17', '2')
    # plot_3_algorithms_corr(LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor(), y1, y2, y3)
    # plot_4_algorithms_corr("LR", "DTR", "RFR","KNN", y1, y2, y3, y4)
    # plot_1_algorithm_corr("LK", y2)
    plot_2_algorithms_corr("LR", "DTR", y1, y2)