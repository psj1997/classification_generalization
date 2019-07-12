from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
print(y)
clf = LogisticRegression(random_state=0, solver='lbfgs',
                       multi_class='multinomial')
clf.fit(X,y)
print(clf.classes_)
pre = clf.predict_proba(X)
print(pre)

