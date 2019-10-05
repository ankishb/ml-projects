import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv('wine.data', header=None)

# given in dataset info file. at ucl_data website
df.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
              'Alcalinity of ash', 'Magnesium', 'Total phenols',
              'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
              'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
              'Proline']

print(df.head())

# Copy all the data in input except class label
x = df.drop('Class label', 1)
y = df['Class label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    random_state=4)

tree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=3,
                              random_state=0)
tree.fit(x_train, y_train)

y_pred = tree.predict(x_test)
print("mis-classified samples %d" % (y_test != y_pred).sum())
print("accuracy : %.2f" %(((y_test == y_pred).sum())/(y_test.shape[0])))

print("accuracy_score %.2f" %(accuracy_score(y_pred, y_test)))
print("class_label", np.unique(y))
print(x_train.shape, x_test.shape)



