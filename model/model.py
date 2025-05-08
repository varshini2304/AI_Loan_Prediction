
import pandas as pd
import numpy as np
import plotnine
from plotnine import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import make_scorer
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore', category = FutureWarning)
df_train = pd.read_csv(
    filepath_or_buffer = 'https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv',
    usecols = [i for i in range(1, 14)]
)
print('Data dimension: {} rows and {} columns'.format(len(df_train), len(df_train.columns)))
df_train.head()
df_test = pd.read_csv(
    filepath_or_buffer = 'https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_test.csv'
)
df_train.info()

df_train = df_train.astype({'Credit_History': object, 'Loan_Status': int})
df_train.select_dtypes(include = ['object']).dtypes
for i in df_train.select_dtypes('object').columns:
    print(df_train[i].value_counts(),'\n')
df_train.isna().sum()
print('Number of missing dependents is about {} rows'.format(df_train['Dependents'].isna().sum()))
df_train['Dependents'].fillna(value = '0', inplace = True) 
print('Number of missing Self_Employed is about {} rows'.format(df_train['Self_Employed'].isna().sum()))
df_train['Self_Employed'].fillna(value = 'No', inplace = True) 
df_train[['Loan_Amount_Term', 'Loan_Status']].groupby('Loan_Status').describe()
print('Percentile 20th: {}'.format(df_train['Loan_Amount_Term'].quantile(q = 0.2)))
df_train['Loan_Amount_Term'].fillna(value = 360, inplace = True)

df_cred_hist = pd.crosstab(df_train['Credit_History'], df_train['Loan_Status'], margins = True).reset_index()

df_cred_hist.columns.name = None

df_cred_hist = df_cred_hist.drop([len(df_cred_hist) - 1], axis = 0)
df_cred_hist.rename(columns = {'Credit_History':'Credit History', 0:'No', 1:'Yes'}, inplace = True)
df_cred_hist

pos_cred_hist0 = df_train[(df_train['Credit_History'].isna()) & (df_train['Loan_Status'] == 0)]
pos_cred_hist1 = df_train[(df_train['Credit_History'].isna()) & (df_train['Loan_Status'] == 1)]
print('Number of rows with Loan_Status is No but Credit_History is NaN  : {}'.format(len(pos_cred_hist0)))
print('Number of rows with Loan_Status is Yes but Credit_History is NaN : {}'.format(len(pos_cred_hist1)))

credit_loan = zip(df_train['Credit_History'], df_train['Loan_Status'])
df_train['Credit_History'] = [
                                0.0 if np.isnan(credit) and status == 0 else
                                1.0 if np.isnan(credit) and status == 1 else
                                credit for credit, status in credit_loan
                             ]
df_train.dropna(axis = 0, how = 'any', inplace = True)

df_train.isna().sum()
df_test.info()

df_test = df_test.astype({'Credit_History': object})
df_test.select_dtypes(include = ['object']).dtypes

for i in df_test.select_dtypes('object').columns:
    print(df_test[i].value_counts(),'\n')
df_test.isna().sum()
print('Number of missing values in Dependents is about {} rows'.format(df_test['Dependents'].isna().sum()))
df_test['Dependents'].fillna(value = '0', inplace = True)
print('Number of missing values in Self_Employed is about {} rows'.format(df_test['Self_Employed'].isna().sum()))
df_test['Self_Employed'].fillna(value = 'No', inplace = True) 
df_test['Loan_Amount_Term'].fillna(value = 360, inplace = True)
df_test.dropna(axis = 0, how = 'any', inplace = True)
df_test.isna().sum()

df_viz_1 = df_train.groupby(['Loan_Status'])['Loan_ID'].count().reset_index(name = 'Total')

df_viz_1['Loan_Status'] = df_viz_1['Loan_Status'].map(
    {
        0: 'Not default',
        1: 'Default'
    }
)

df_viz_1

plt.figure(figsize = (6.4,4.8))


colors = ['#80797c','#981220']


explode = (0.1, 0)


plt.pie(
    x = 'Total',
    labels = 'Loan_Status',
    data = df_viz_1,
    explode = explode,
    colors = colors,
    autopct = '%1.1f%%',
    shadow = False,
    startangle = 140
)


plt.title('Number of customers by loan status', fontsize = 18)
plt.axis('equal')
plt.show()

df_viz_2 = df_train.groupby(['Loan_Status', 'Dependents'])['Loan_ID'].count().reset_index(name = 'Total')

df_viz_2['Loan_Status'] = df_viz_2['Loan_Status'].map(
    {
        0: 'Not default',
        1: 'Default'
    }
)

df_viz_2
plotnine.options.figure_size = (8, 4.8)
(
    ggplot(
        data = df_viz_2
    )+
    geom_bar(
        aes(
            x = 'Dependents',
            y = 'Total',
            fill = 'Loan_Status'
        ),
        stat = 'identity',
        position = 'fill',
        width = 0.5
    )+
    labs(
        title = 'The composition of loan status by the dependents',
        fill = 'Loan status'
    )+
    xlab(
        'Dependents'
    )+
    ylab(
        'Frequency'
    )+
    scale_x_discrete(
        limits = ['0', '1', '2', '3+']
    )+
    scale_fill_manual(
        values = ['#981220','#80797c'],
        labels = ['Default', 'Not Default']
    )+
    theme_minimal()
)

df_viz_3 = df_train.groupby(['Loan_Status', 'Education'])['Loan_ID'].count().reset_index(name = 'Total')

df_viz_3['Loan_Status'] = df_viz_3['Loan_Status'].map(
    {
        0: 'Not default',
        1: 'Default'
    }
)
df_viz_3
plotnine.options.figure_size = (8, 4.8)
(
    ggplot(
        data = df_viz_3
    )+
    geom_bar(
        aes(
            x = 'Education',
            y = 'Total',
            fill = 'Loan_Status'
        ),
        stat = 'identity',
        position = 'fill',
        width = 0.5
    )+
    labs(
        title = 'The composition of loan status by the education',
        fill = 'Loan status'
    )+
    xlab(
        'Educations'
    )+
    ylab(
        'Frequency'
    )+
    scale_x_discrete(
        limits = ['Graduate', 'Not Graduate']
    )+
    scale_fill_manual(
        values = ['#981220','#80797c'],
        labels = ['Default', 'Not Default']
    )+
    theme_minimal()
)

df_viz_4 = df_train[['ApplicantIncome', 'Loan_Status']].reset_index(drop = True)

df_viz_4['Loan_Status'] = df_viz_4['Loan_Status'].map(
    {
        0: 'Not default',
        1: 'Default'
    }
)
df_viz_4.head()
plotnine.options.figure_size = (8, 4.8)
(
    ggplot(
        data = df_viz_4
    )+
    geom_density(
        aes(
            x = 'ApplicantIncome',
            fill = 'Loan_Status'
        ),
        color = 'white',
        alpha = 0.85
    )+
    labs(
        title = 'The distribution of applicant incomes by loan status'
    )+
    scale_fill_manual(
        name = 'Loan Status',
        values = ['#981220','#80797c'],
        labels = ['Default', 'Not Default']
    )+
    xlab(
        'Applicant income'
    )+
    ylab(
        'Density'
    )+
    theme_minimal()
)

df_viz_5 = df_train[['LoanAmount', 'Loan_Status']].reset_index(drop = True)

df_viz_5['Loan_Status'] = df_viz_5['Loan_Status'].map(
    {
        0: 'Not default',
        1: 'Default'
    }
)
df_viz_5.head()
plotnine.options.figure_size = (8, 4.8)
(
    ggplot(
        data = df_viz_5
    )+
    geom_density(
        aes(
            x = 'LoanAmount',
            fill = 'Loan_Status'
        ),
        color = 'white',
        alpha = 0.85
    )+
    labs(
        title = 'The distribution of loan amount by loan status'
    )+
    scale_fill_manual(
        name = 'Loan Status',
        values = ['#981220','#80797c'],
        labels = ['Default', 'Not Default']
    )+
    xlab(
        'Loan amount'
    )+
    ylab(
        'Density'
    )+
    theme_minimal()
)

df_test['Loan_Status'] = 999

df_concat = pd.concat(objs = [df_train , df_test], axis = 0)

df_concat.drop(columns = ['Loan_ID'], inplace = True)

cols_obj_train = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
print(cols_obj_train)
df_concat = pd.get_dummies(data = df_concat, columns = cols_obj_train, drop_first = True)
print('Dimension data: {} rows and {} columns'.format(len(df_concat), len(df_concat.columns)))
df_concat.head()

df_concat['Loan_Status'].value_counts()

df_train = df_concat[df_concat['Loan_Status'].isin([0, 1])].reset_index(drop = True)
print('Dimension data: {} rows and {} columns'.format(len(df_train), len(df_train.columns)))
df_train.head()
df_test = df_concat[df_concat['Loan_Status'].isin([999])].reset_index(drop = True)
print('Data dimension: {} rows and {} columns'.format(len(df_test), len(df_test.columns)))
df_test.head()


df_train_final = df_train.reset_index(drop = True)
X = df_train_final[df_train_final.columns[~df_train_final.columns.isin(['Loan_Status'])]]
y = df_train_final['Loan_Status']

X_train, X_val, y_train, y_val = train_test_split(X , y, test_size = 0.3, random_state = 42)
print('Data dimension of training set   :', X_train.shape)
print('Data dimension of validation set :', X_val.shape)

X_test = df_test[df_test.columns[~df_test.columns.isin(['Loan_Status'])]]
print('Data dimension of testing set    :', X_test.shape)
xgb_model = xgb.XGBClassifier(
    objective = 'binary:logistic',
    use_label_encoder = False
)

params = {
    'eta': np.arange(0.1, 0.26, 0.05),
    'min_child_weight': np.arange(1, 5, 0.5).tolist(),
    'gamma': [5],
    'subsample': np.arange(0.5, 1.0, 0.11).tolist(),
    'colsample_bytree': np.arange(0.5, 1.0, 0.11).tolist()
}

scorers = {
    'f1_score': make_scorer(f1_score),
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score)
}


skf = KFold(n_splits = 10, shuffle = True)

grid = GridSearchCV(
    estimator = xgb_model,
    param_grid = params,
    scoring = scorers,
    n_jobs = -1,
    cv = skf.split(X_train, np.array(y_train)),
    refit = 'accuracy_score'
)
grid.fit(X = X_train, y = y_train)
grid.best_params_
predicted = grid.predict(X_val)

accuracy_baseline = accuracy_score(predicted, np.array(y_val))
recall_baseline = recall_score(predicted, np.array(y_val))
precision_baseline = precision_score(predicted, np.array(y_val))
f1_baseline = f1_score(predicted, np.array(y_val))

print('Accuracy for baseline   :{}'.format(round(accuracy_baseline, 5)))
print('Recall for baseline     :{}'.format(round(recall_baseline, 5)))
print('Precision for baseline  :{}'.format(round(precision_baseline, 5)))
print('F1 Score for baseline   :{}'.format(round(f1_baseline, 5)))
filename = 'C:\\Users\\varsh\\projects\\Loan_approval_ML\\model\\loan_model.pkl'
joblib.dump(grid.best_estimator_, filename)