import math, datetime, time
import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import quandl
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')
df = quandl.get('NSE/IDEA')
# print (df.head())

df = df[['Open', 'High', 'Low', 'Close', 'Total Trade Quantity']]
df['Hl_PCT'] = (df['High'] - df['Close']) / df['Close'] * 100
df['CHANGE'] = (df['Close'] - df['Open']) / df['Open'] * 100

# print (df.head())

forecast = 'Close'
df.fillna('-99999', inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
# print(forecast_out)
df["label"] = df[forecast].shift(-forecast_out)

x = np.array(df.drop(['label'], 1))
x = preprocessing.scale(x)
x = x[:-forecast_out]
x_lately = x[-forecast_out:]

# x = x[:-forecast_out + 1]
df.dropna(inplace=True)
# y = np.array(df['label'])
y = np.array(df['label'])
# print (len(x), len(y))
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(x, y, test_size=0.2)
clf = LinearRegression()
# clf = svm.SVR()
clf.fit(X_train, Y_train)
with open('Linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

pickle_in = open('Linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)
accuracy = clf.score(X_test, Y_test)

# print(accuracy)
forecast_set = clf.predict(x_lately)
print(forecast_set, accuracy, forecast_out)
df['forecast'] = np.nan

lastdate = df.iloc[-1].name
print (lastdate)
lastUnix = time.mktime(lastdate.timetuple())
oneday = 86400
nextUnix = lastUnix + oneday

for i in forecast_set:
    nextDate = datetime.datetime.fromtimestamp(nextUnix)
    nextUnix += oneday
    df.loc[nextDate] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Close'].plot()
df['forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
