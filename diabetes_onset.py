import pandas as pd
import numpy as np
import keras

# import the uci pima indians diabetes dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['n_pregnant', 'glucose_concentration', 'blood_pressuer (mm Hg)', 'skin_thickness (mm)', 'serum_insulin (mu U/ml)',
        'BMI', 'pedigree_function', 'age', 'class']

df = pd.read_csv(url, names = names)
df[df['glucose_concentration'] == 0]

columns = ['glucose_concentration',
           'blood_pressuer (mm Hg)', 'skin_thickness (mm)', 'serum_insulin (mu U/ml)', 'BMI']

for col in columns:
    df[col].replace(0, np.NaN, inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)
dataset = df.values
print(dataset.shape)

X = dataset[:,0:8]
y = dataset[:,8].astype(int)

# Normalize the data using sklearn StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)