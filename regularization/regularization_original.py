import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

covid = pd.read_csv(r'ITA_covid_model_1.csv')
#print(covid)
print("sample_data statistics:", covid.describe())
ax = sns.regplot(x="hosp_patients", y="new_deaths", data=covid, fit_reg=True)
ax = sns.regplot(x="icu_patients", y="new_deaths", data=covid, fit_reg=True)

ax.set(xscale="log")
plt.xticks(rotation=90)
plt.tight_layout()

sns.jointplot(y="icu_patients", x="hosp_patients", data=covid, kind="reg")

plt.xticks(rotation=90)
plt.show()
plt.show()