# evaluate an lasso regression model on the dataset
from numpy import mean
from numpy import std
from numpy import absolute
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import ResidualsPlot

# load the dataset
data_full = pd.read_csv(r'UK_covid_model_1.csv')
#data = data_full[["total_cases", "new_cases", "new_deaths", "icu_patients"]]

data = data_full[["total_cases", "new_cases", "new_deaths", "icu_patients", "hosp_patients", "new_tests", "total_tests", "tests_per_case"]]
data_full.replace('', 0)
data_full.fillna(0)
#view first six rows of data
print(data[0:6])

#define predictor and response variables
X = data[["total_cases", "new_cases", "icu_patients", "hosp_patients", "new_tests", "total_tests",  "tests_per_case"]]
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

new_1 = [3846851,	18668,	3726,	32945,	643204, 71011933,	27.5]
new_2 = [3863757,	16906,	3638,	31750,	606382,	71642534,	28.1]
new_3 = [3882972,	19215,	3628,	30584,	801949,	72464146,	29.5]
new_4 = [3903706,	20734,	3572,	29388,	783851,	73277874,	31.3]
new_5 = [3922910,	19204,	3505,	28352,	671585,	73971219,	33.6]
new_6 = [3941273,	18363,	3373,	26880,	454008,	74437279,	35]
new_7 = [3957177,	15904,	3302,	26765,	584933,	75043566,	36.5]
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
