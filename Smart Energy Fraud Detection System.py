# electricity theft detection mini project
# Samagra Gune - 25BCE10462

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(42)

# making fake data since i dont have actual dataset
# normal users use around 15 units, theft users are much lower and random

all_users = []

for i in range(300):
    
    if i <= 209:
        usage = np.random.normal(15, 2.5, 30)
        lbl = 0
    else:
        usage = np.random.normal(4, 6, 30)
        # sometimes meter shows negative if bypassed
        if np.random.random() < 0.3:
            usage[np.random.randint(0, 30)] = np.random.uniform(-6, -1)
        lbl = 1

    # features - mean std min and how many days had zero
    m = usage.mean()
    s = usage.std()
    mn = usage.min()
    z = int(np.sum(usage < 0.5))  # tried == 0 first but floating point was annoying

    all_users.append([m, s, mn, z, lbl])

df = pd.DataFrame(all_users, columns=['mean_use', 'std_use', 'min_use', 'low_days', 'target'])

# quick check
print(df.shape)
print(df['target'].value_counts())
print(df.describe())

X = df[['mean_use', 'std_use', 'min_use', 'low_days']]
y = df['target']

# 70-30 split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)

# logistic regression first as baseline
lr = LogisticRegression()
lr.fit(Xtrain, ytrain)
lr_pred = lr.predict(Xtest)
print("LR acc:", accuracy_score(ytest, lr_pred))

# random forest - this one worked better
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(Xtrain, ytrain)
rf_pred = rf.predict(Xtest)
print("RF acc:", accuracy_score(ytest, rf_pred))

# confusion matrix
print(confusion_matrix(ytest, lr_pred))
print(confusion_matrix(ytest, rf_pred))

# checking which features matter
feat_names = ['mean_use', 'std_use', 'min_use', 'low_days']
for f, imp in zip(feat_names, rf.feature_importances_):
    print(f, round(imp, 3))

# plot - blue = normal, red = theft
c = ['blue' if t == 0 else 'red' for t in df['target']]
plt.scatter(df['mean_use'], df['std_use'], c=c, alpha=0.5)
plt.xlabel('Mean Usage')
plt.ylabel('Std Dev')
plt.title('Theft Detection Plot')
plt.show()
