from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import math
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

#Classifiers
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

def cross_fold_validation(X, Ylist, clf):
    f1_score = 0
    mcc_score = 0
    kf = StratifiedKFold(n_splits=10, random_state=True, shuffle=True)
    for train_index, test_index in kf.split(X, Ylist):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Ylist[train_index], Ylist[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        mcc_score = metrics.matthews_corrcoef(y_test, y_pred) + mcc_score
        f1_score = metrics.f1_score(y_test, y_pred, average='macro') + f1_score

    f1_score = f1_score / 10
    mcc_score = mcc_score / 10
    return f1_score, mcc_score

def gridsearch_params(X,Ylist):
    # 'gamma':[1, 0.1, 0.01, 0.001, 0.0001, 0]
    n_range = range(1, 101)
    m_range = range(1, 51)
    max_depth=range(1, 5)
    min_leaf=range(1, 11)
    loss=[True, False]
    presort=[True, False]
    #n_estimators=n_range, criterion=criterion, min_samples_split=split_range, max_depth=max_depth, min_samples_leaf=min_leaf
    parameters = dict(verbose=presort)

    model = svm.SVC(C = 28, kernel = 'rbf', gamma=0.001, probability=True)
    grid = GridSearchCV(model, parameters, n_jobs=-1, scoring='f1_macro')
    grid.fit(X, Ylist)

    print(grid.best_score_)
    print(grid.best_estimator_)

def __mcc_score(tp, tn, fp, fn):
    return (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

def hold_out_confusion_matrix(X, Ylist, XWild, YWild, clf):
    clf.fit(X, Ylist)
    y_pred = clf.predict(XWild)
    tn, fp, fn, tp = confusion_matrix(YWild, y_pred).ravel()
    return tn, fp, fn, tp

# Train test split
def hold_out_wild(X, Ylist, XWild, YWild, clf):  # Hold-out test
    clf.fit(X, Ylist)
    y_pred = clf.predict(XWild)
    score_f1 = metrics.f1_score(YWild, y_pred, average='macro')
    mcc_score = metrics.matthews_corrcoef(YWild, y_pred)
    return score_f1, mcc_score

def hold_out_test(X, Ylist, clf):  # Hold-out test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Ylist, random_state=5)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    score_f1 = metrics.f1_score(Y_test, y_pred, average='macro')
    mcc_score = metrics.matthews_corrcoef(Y_test, y_pred)

    return score_f1, mcc_score