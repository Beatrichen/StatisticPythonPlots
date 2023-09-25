# evaluate an lasso regression model on the dataset
from numpy import mean
from numpy import std
from numpy import absolute
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import ResidualsPlot

# load the dataset
data_full = pd.read_csv(r'ITA_covid_model_3.csv')
#data = data_full[["total_cases", "new_cases", "new_deaths", "icu_patients"]]

data = data_full[["new_cases", "new_deaths", "icu_patients", "hosp_patients", "new_tests", "tests_per_case"]]
data_full.replace('', 0)
data_full.fillna(0)
#view first six rows of data
print(data[0:6])

#define predictor and response variables
X = data[["new_cases", "icu_patients", "hosp_patients", "new_tests", "tests_per_case"]]
y = data["new_deaths"]

# define model
model = Lasso(alpha=1.0)
visualizer = ResidualsPlot(model)
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
print(X_train.shape); print(X_test.shape)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))

model.fit(X, y)

r_sq = model.score(X, y)
print('r_sq', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

# define new data

new_1 = [22210, 2553,	25375,	157524,	8.4]
new_2 = [11825,	2569,	25517,	67174,	8.2]
new_3 = [14245,	2583,	25658,	102974,	8.2]
new_4 = [10798,	2579,	25896,	77993,	8.1]
new_5 = [15375,	2569,	25964,	135106,	7.8]
new_6 = [20326,	2571,	25745,	178596,	7.7]
new_7 = [18416,	2587,	25878,	121275,	7.4]
# make a prediction
yhat_1 = model.predict([new_1])
yhat_2 = model.predict([new_2])
yhat_3 = model.predict([new_3])
yhat_4 = model.predict([new_4])
yhat_5 = model.predict([new_5])
yhat_6 = model.predict([new_6])
yhat_7 = model.predict([new_7])
# summarize prediction
print('Predicted: %.3f' % yhat_1)
print('Predicted: %.3f' % yhat_2)
print('Predicted: %.3f' % yhat_3)
print('Predicted: %.3f' % yhat_4)
print('Predicted: %.3f' % yhat_5)
print('Predicted: %.3f' % yhat_6)
print('Predicted: %.3f' % yhat_7)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()