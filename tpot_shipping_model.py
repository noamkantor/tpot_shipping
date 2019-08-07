from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd

#load data -- change file name if needed
df = pd.read_csv(r'predictions_8_6.csv')
df = df[df['Shipping'] != 'Unknown']

#even out classes
df2 = df[df['Shipping'] == '1']
df2 = df2.sample(n=3492)
df = df[df['Shipping'] == '0']
df = pd.concat([df, df2])

#move to numpy
target = df['Shipping'].values
df_selection = df.loc[:, 'SVCS':'commercial_L_area']
data = df_selection.values

#split data
X_train, X_test, y_train, y_test = train_test_split(data.astype(np.float64),
    target.astype(np.float64), train_size=0.75, test_size=0.25)

#train and compare classifiers
tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_iris_pipeline.py')

#evaluate (code by Raanan)
y_pred_class = tpot.predict(X_test)
auc = metrics.roc_auc_score(y_test, y_pred_class)
print("AUC Score: ", auc)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_class)
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

print('Confusion matrix')
print(metrics.confusion_matrix(y_test, y_pred_class)) # confusion matrix
print('Classification report')
print(metrics.classification_report(y_test, y_pred_class))