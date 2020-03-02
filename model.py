from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics, preprocessing
from matplotlib import pyplot as plt
from createCSV import finalize_csv
import pandas as pd
import numpy as np
import sys

scaler = MinMaxScaler()

cols = ['DATE', 'HOME', 'VISITOR', 'VISITOR_PTS', 'HOME_PTS', 'WINNER', 'HOME_B2B', 'VISITOR_B2B',  
        'HOME_LAST_SEASON_W%', 'VISITOR_LAST_SEASON_W%', 'HOME_GAME', 'AWAY GAME']

df_2018 = pd.read_csv("season_2018.csv")
df_2019 = pd.read_csv("season_2019.csv")
df_2020 = pd.read_csv(finalize_csv('season_2020.csv', 2020))
df = pd.concat([df_2018, df_2019], ignore_index=True)

columns = ['HOME_NRtg', 'HOME_DRB%', 'HOME_SRS', 'VISITOR_NRtg', 
    'VISITOR_DRB%', 'VISITOR_SRS', 'HOME_ORtg', 'HOME_DRtg', 'VISITOR_ORtg', 'VISITOR_DRtg',
    'HOME_GAME', 'AWAY GAME', 'VISITOR_MOV', 'HOME_MOV', 'HOME_PACE', 'VISITOR_PACE']
df[columns] = scaler.fit_transform(df[columns])
df_2020[columns] = scaler.fit_transform(df_2020[columns])

X = df.drop(columns=cols)
y = df['WINNER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=24)

def SVMModel(print_bool):
    # Test Accuracy : 70.3%
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if print_bool:
        scores = cross_val_score(clf, X, y, cv=10)

        sys.stderr.write("[K-FOLD ACCURACY] " + str(scores.mean()) + " (+/- " + str(scores.std()) + ")\n")
        sys.stderr.write("[ACCURACY] " + str(metrics.accuracy_score(y_test, y_pred)) + "\n")
        sys.stderr.write("[F1 SCORE] " + str(metrics.f1_score(y_pred, y_test)) + "\n")
        sys.stderr.flush()
    predict_results = df_2020.drop(columns=cols)
    return clf.predict_proba(predict_results)

average = SVMModel(True)
average = pd.DataFrame(average, columns=[['VISITOR_WIN_PROB', 'HOME_WIN_PROB']])
sys.stderr.write("[Progress] 1/100 runs completed.\n")
sys.stderr.flush()

for n in range(0, 99):
    nth_run = SVMModel(False)
    nth_run = pd.DataFrame(nth_run, columns=[['VISITOR_WIN_PROB', 'HOME_WIN_PROB']])
    average = average.add(nth_run).div(2)
    if (n+2) % 10 == 0:
        sys.stderr.write("[Progress] " + str(n+2) + "/100 runs completed.\n")
        sys.stderr.flush()

df_2020 = pd.concat([df_2020, average], axis=1, sort=True)
df_2020.to_csv('season_2020.csv', index=False)
print("[COMPLETE] CSV file update completed!")
