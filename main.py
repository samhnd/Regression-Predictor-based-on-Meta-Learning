from automl import prediction
import pandas as pd
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')
print('start')

df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
              'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
cancer_ds = load_breast_cancer()
cancer_df = pd.DataFrame(cancer_ds.data, columns=cancer_ds.feature_names)
cancer_df['target'] = cancer_ds.target

a = prediction(cancer_df, 'target')
