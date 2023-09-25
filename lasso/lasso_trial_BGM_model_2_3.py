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
data_full = pd.read_csv(r'BGM_covid_model_2.csv')
#data = data_full[["total_cases", "new_cases", "new_deaths", "icu_patients"]]

data = data_full[["new_cases", "new_deaths", "icu_patients", "hosp_patients", "new_tests","tests_per_case"]]
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

new_1 = [1793,	489,	2081,	11933,	13.7]
new_2 = [880,	481,	2122,	18577,	13.9]
new_3 = [842,	468,	2185,	17637,	14.1]
new_4 = [876,	464,	2226,	40229,	14.9]
new_5 = [1848,	430,	2130,	60064,	16.1]
new_6 = [2997,	404,	2076,	55217,	16.9]
new_7 = [2923,	381,	2018,	44353,	17.5]
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
