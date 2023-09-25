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
data_full = pd.read_csv(r'Europe-2020-prediction.csv')
#data = data_full[["total_cases", "new_cases", "new_deaths", "icu_patients"]]

data = data_full[["total_cases", "new_cases", "new_deaths", "icu_patients", "hosp_patients", "new_tests", "total_tests","tests_per_case"]]
data_full.replace('', 0)
data_full.fillna(0)
#view first six rows of data
print(data[0:6])

#define predictor and response variables
X = data[["total_cases", "new_cases", "icu_patients", "hosp_patients", "new_tests", "total_tests", "tests_per_case"]]
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

# fit the model
model.fit(X, y)

r_sq = model.score(X, y)
print('r_sq', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

# define new data
new_1 = [24016698,	209555,	18233,	132288,	1319974,	243120878,	472]
new_2 = [24176189,	159491, 18503,	134286,	1053459,	148001477,	477.3]
new_3 = [24344231,	168042, 18791,	137532,	913397,	291970856,	491.6]
new_4 = [24547995,	203764, 21215,	155268,	1837171,	245867845,	486.8]
new_5 = [24823686,	275691, 21176,	155435,	2022119,	253482732,	487.4]
new_6 = [25093278,	269592, 18962,	141713,	2088148,	254643897,	500.9]
new_7 = [25414033,	320755, 21379,	156528,	2096082,	278251171,	505.5]

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
