# imports:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.formula.api as smf

from sklearn.preprocessing import StandardScaler,  MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, classification_report, roc_curve, auc

'''
Data Dictionary:

Pregnancies: Number of times pregnant.
Glucose: Plasma glucose concentration.
BloodPressure: blood pressure (mm Hg).
SkinThickness: Triceps skin fold thickness (mm).
Insulin: 2-Hour serum insulin (mu U/ml).
BMI: Body mass index (weight in kg/(height in m)^2).
DiabetesPedigreeFunction: Diabetes pedigree function.
Age: (years).
Outcome: Class variable (0 or 1), Class Distribution: (class value 1 is interpreted as "tested positive for diabetes")
'''

# Functions
def remove_high_outliers(df, feature, max) :
    return df.drop(df[df[feature] > max].index)

def remove_low_outliers(df, feature, min) :
    return df.drop(df[df[feature] < min].index)

# Importing the CSV here:
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv')
df_raw.to_csv('../data/raw/diabetes_raw.csv')

print(f'The Dataset has {df_raw.shape[0]} "observations" with {df_raw.shape[1]} columns')
print(f'There are NO Nulls')
print(f'All features are numerical')

'''
Conclusions:

Pregnancies: remove > 12
Glucose: replace 0 -> median, group 10 [30_200]
BlodPressure: replace 0-> median, remove > 100, group 10 [20_100]
SkinThickness: replace 0 -> median, remove > 60, group 5 [0_60]
Insulin: remove > 300 ?, group 10 [0_300] 
BMI: replace 0 -> median, remoove > 50, grouṕ 5 [15_50]
DiabetesPedigree: remove > 1.2, group 0.100 [0_1.200]
Age: remove > 70, group 10 [20_70]
'''

df_interim = df_raw.copy()

# Remove High Outliers:
df_interim = remove_high_outliers(df_interim, 'Pregnancies', max=12)
df_interim = remove_high_outliers(df_interim, 'BloodPressure', max=100)
df_interim = remove_high_outliers(df_interim, 'SkinThickness', max=60)
#df_interim = remove_outliers(df_interim, 'Insulin', max=300)
df_interim = remove_high_outliers(df_interim, 'BMI', max=50)
#df_interim = remove_outliers(df_interim, 'DiabetesPedigreeFunction', max=1.2)
df_interim = remove_high_outliers(df_interim, 'Age', max=70)

# replace 0's for median or mean:
features_replace_values = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
for feature in features_replace_values :
    # median:
    # df_interim[feature] = df_interim[feature].mask(df_interim[feature] == 0, df_interim[feature].median())
    # mean:
    df_interim[feature] = df_interim[feature].mask(df_interim[feature] == 0, df_interim[feature].mean())

# Remove Low Outliers:
df_interim = remove_low_outliers(df_interim, 'BloodPressure', min=40)

df_interim.to_csv('../data/interim/diabetes_interim.csv')
df = df_interim.copy()

# Separation: features & target
X = df.drop('Outcome', axis='columns')
Y = df["Outcome"]

# Scale the features --> BEST RESULTS WERE ACHIEVED WITH NO SCALATION!!!
'''
scaler = MinMaxScaler()
scaler.fit_transform(X)
'''

# train test split:
X_train,X_test,Y_train,Y_test=train_test_split(X, Y, stratify=Y,  test_size=0.3)


# Model & Fit:
model_dec_tree = DecisionTreeClassifier()
model_dec_tree.fit(X_train,Y_train)

print(f'='*30)
print(f'STANDAR DECISION TREE RESULTS & METRICS')
print(f'='*30)

# Get the score of train data just to verify its 1.
score = model_dec_tree.score(X_train, Y_train)
print(f'The score for Decision Tree with X_train & Y_trains is: {score}')

#Get the score for the predictions:
score = model_dec_tree.score(X_test, Y_test)
print(f'The score for Decision Tree with X_test & Y_test is: {score}')

# Accuracy:
#acc_score = print(accuracy_score(Y_test, model_dec_tree.predict(X_test)))
#print(f'The accuracy for Decision Tree (entropy) with X_test is: {acc_score}')

# Confusion Matrix:
print(confusion_matrix(Y_test, model_dec_tree.predict(X_test)))
sns.heatmap(confusion_matrix(Y_test, model_dec_tree.predict(X_test)), annot=True)
plt.show()

# classification report:
print(classification_report(Y_test, model_dec_tree.predict(X_test)))

# feature importance:
print(f'Features importance: \n {model_dec_tree.feature_importances_}')

# Get the Number of Leaves
print(f'Number of leaves: {model_dec_tree.get_n_leaves()}')

# Tree params
print(f'Tree params: \n {model_dec_tree.get_params()}')

# Tree depth
print(f'Tree depth: {model_dec_tree.get_depth()}')


# Grid Search
dt_parms = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150],'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
clf = GridSearchCV(DecisionTreeClassifier(), dt_parms, cv=5)
clf.fit(X_train, Y_train)
#print(clf.best_params_)
#print(clf.best_estimator_)
estimator = clf.best_estimator_
print(f'BEST HYPERPARAMETERS:')
print(f'criterion: {estimator.criterion}')
print(f'max_depth: {estimator.max_depth}')
print(f'min_samples_split: {estimator.min_samples_split}')

'''
BEST HYPERPARAMETERS:
criterion: entropy
max_depth: 4
min_samples_split: 3
'''

# Train with Best Hyperparams
model_best_dt = DecisionTreeClassifier(criterion=estimator.criterion, max_depth=estimator.max_depth, min_samples_split=estimator.min_samples_split)
model_best_dt.fit(X_train, Y_train)

# Score for train data:
score = model_best_dt.score(X_train, Y_train)
print(f'The score for Decision Tree (entropy) with X_train & Y_trains is: {score}')


print(f'='*30)
print(f'TUNNED DECISION TREE RESULTS & METRICS')
print(f'='*30)

# Score for the predictions:
score = model_best_dt.score(X_test, Y_test)
print(f'The score for Decision Tree (entropy) with X_test & Y_test is: {score}')

# Accuracy:
#acc_score = print(accuracy_score(Y_test, model_best_dt.predict(X_test)))
#print(f'The accuracy for Decision Tree (entropy) with X_test is: {acc_score}')

# Confusion Matrix:
print(confusion_matrix(Y_test, model_best_dt.predict(X_test)))
sns.heatmap(confusion_matrix(Y_test, model_best_dt.predict(X_test)), annot=True)
plt.show()

# Classification report:
print(classification_report(Y_test, model_best_dt.predict(X_test)))

# Features importance:
print(f'Features importance: \n {model_best_dt.feature_importances_}')

# Number of leaves:
print(f'Number of leaves: {model_best_dt.get_n_leaves()}')

# Tree params:
print(f'Tree params: \n {model_best_dt.get_params()}')

# Tree depth:
print(f'Tree depth: {model_best_dt.get_depth()}')

#Measure results ROC AUC
fpr, tpr, thresholds = roc_curve(Y_test, model_best_dt.predict_proba(X_test)[:,1])
plt.figure(figsize=(8,8))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.plot([0,1],[0,1],color="navy",lw=2,label="Random-Model")
plt.plot(fpr,tpr,color="darkorange",lw=2, label="Decision Tree- Model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic: ROC-AUC")
plt.legend()
plt.show()
print("Computed Area Under the Curve (AUC)", auc(fpr, tpr))