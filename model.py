from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

scaler = MinMaxScaler()

df = pd.read_csv("season_2020.csv")
columns = ['HOME_NRtg', 'HOME_DRB%', 'HOME_SRS', 'VISITOR_NRtg', 'VISITOR_DRB%', 'VISITOR_SRS']
df[columns] = scaler.fit_transform(df[columns])

games_played = df['WINNER'].count()
games_not_played = len(df.index) - games_played

cols = ['DATE', 'VISITOR', 'VISITOR_PTS', 'HOME', 'HOME_PTS', 'WINNER']

X = df.drop(columns=cols).head(games_played)
y = df['WINNER'].head(games_played)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=24)

def kerasModel():
    # Test Accuracy: ~69%
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(18,)),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ])

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=100, batch_size=1)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)

    predict_results = df.drop(columns=cols).tail(games_not_played)
    print(model.predict(predict_results))

def SVCModel():
    # Test Accuracy : 72.7%
    clf = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Test accuracy:", metrics.accuracy_score(y_test, y_pred))
    predict_results = df.drop(columns=cols).tail(games_not_played)
    print(clf.predict_proba(predict_results))

def LDAModel():
    # Test Accuracy: 72.35%
    model = LDA(n_components=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Test accuracy:", metrics.accuracy_score(y_test,y_pred))

SVCModel()