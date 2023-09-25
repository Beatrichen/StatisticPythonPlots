import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
import plotly.graph_objects as go

# Create dataframe using IRIS dataset
df = pd.read_csv(r'ITA_covid_model_2_scatter.csv')
df.fillna(0)


# Create pairplot of all the variables with hue set to class
sns.pairplot(df, corner=True)
plt.tight_layout()
plt.show()