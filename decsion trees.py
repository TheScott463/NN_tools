import numpy as np
from sklearn.model_selection import KFold

X = ["a", "b", "c", "d"]
kf = KFold(n_splits=2)
for train, test in kf.split(X):
    print("%s %s" % (train, test))

print("Now 2nd")
import numpy as np
from sklearn.model_selection import RepeatedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
random_state = 12883823
rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
for train, test in rkf.split(X):
    print("%s %s" % (train, test))

print("Now 3rd")
from sklearn.model_selection import LeaveOneOut
X = [1, 2, 3, 4,-1,5,1,7,0,-12,-5,19,1024,0,0,0,0,0,1,3]
loo = LeaveOneOut()
for train, test in loo.split(X):
    print(f"{train} {test}")
exit()

from sklearn.metrics.scorer import make_scorer
scoring = {'prec_macro': 'precision_macro','rec_micro': make_scorer(recall_score, average='macro')}
scores = cross_validate(clf, iris.data, iris.target, scoring=scoring,cv=5, return_train_score=True)
sorted(scores.keys())
['fit_time', 'score_time', 'test_prec_macro', 'test_rec_micro',
 'train_prec_macro', 'train_rec_micro']
scores['train_rec_micro']
# array([0.97..., 0.97..., 0.99..., 0.98..., 0.98...])

scores = cross_validate(clf, iris.data, iris.target,scoring='precision_macro', cv=5,return_estimator=True)
sorted(scores.keys())
['estimator', 'fit_time', 'score_time', 'test_score', 'train_score']


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

from sklearn.pipeline import make_pipeline
clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
cross_val_score(clf, iris.data, iris.target, cv=cv)



iris = datasets.load_iris()

iris.data.shape, iris.target.shape\
    ((150, 4), (150,))

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

X_train.shape, y_train.shape((90, 4), (90,))
X_test.shape, y_test.shape
((60, 4), (60,))

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)

exit()

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# Function importing Dataset
def importdata():
    balance_data = pd.read_csv (
        'https://archive.ics.uci.edu/ml/machine-learning-' +
        'databases/balance-scale/balance-scale.data',
        sep=',', header=None)

    # Printing the dataswet shape
    print ("Dataset Lenght: ", len (balance_data))
    print ("Dataset Shape: ", balance_data.shape)

    # Printing the dataset obseravtions
    print ("Dataset: ", balance_data.head ())
    return balance_data


# Function to split the dataset
def splitdataset(balance_data):
    # Seperating the target variable
    X = balance_data.values[:, 1:5]
    Y = balance_data.values[:, 0]

    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split (
        X, Y, test_size=0.3, random_state=100)

    return X, Y, X_train, X_test, y_train, y_test


# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier (criterion="gini",
                                       random_state=100, max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_gini.fit (X_train, y_train)
    return clf_gini


# Function to perform training with entropy.
def tarin_using_entropy(X_train, X_test, y_train):
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier (
        criterion="entropy", random_state=100,
        max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_entropy.fit (X_train, y_train)
    return clf_entropy


# Function to make predictions
def prediction(X_test, clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict (X_test)
    print ("Predicted values:")
    print (y_pred)
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    print ("Confusion Matrix: ",
           confusion_matrix (y_test, y_pred))

    print ("Accuracy : ",
           accuracy_score (y_test, y_pred) * 100)

    print ("Report : ",
           classification_report (y_test, y_pred))


# Driver code
def main():
    # Building Phase
    data = importdata ()
    X, Y, X_train, X_test, y_train, y_test = splitdataset (data)
    clf_gini = train_using_gini (X_train, X_test, y_train)
    clf_entropy = tarin_using_entropy (X_train, X_test, y_train)

    # Operational Phase
    print ("Results Using Gini Index:")

    # Prediction using gini
    y_pred_gini = prediction (X_test, clf_gini)
    cal_accuracy (y_test, y_pred_gini)

    print ("Results Using Entropy:")
    # Prediction using entropy
    y_pred_entropy = prediction (X_test, clf_entropy)
    cal_accuracy (y_test, y_pred_entropy)


# Calling main function
if __name__ == "__main__":
    main ()



# broken code below
exit()

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
iris = load_iris()
cross_val_score(clf, iris.data, iris.target, cv=10)

# rray([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
#         0.93...,  0.93...,  1.     ,  0.93...,  1.      ])





exit()

import pandas as pd
from sklearn import tree
from sklearn.datasets import load_iris
import numpy as np
from IPython.display import Image
import pydotplus


train_file='train_RUN.csv'
train=pd.read_csv(train_file)

#impute number values and missing values
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"] = train["Embarked"].fillna("S")
train["Embarked"][train["Embarked"] == "S"]= 0
train["Embarked"][train["Embarked"] == "C"]= 1
train["Embarked"][train["Embarked"] == "Q"]= 2
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Pclass"] = train["Pclass"].fillna(train["Pclass"].median())
train["Fare"] = train["Fare"].fillna(train["Fare"].median())

target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare","SibSp","Parch","Embarked"]].values


# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)

iris=load_iris()

my_tree_one = my_tree_one.fit(features_one, target)

tree.export_graphviz(my_tree_one, out_file='tree.dot')




def give_nodes(nodes,amount_of_branches,left,right):
    amount_of_branches*=2
    nodes_splits=[]
    for node in nodes:
        nodes_splits.append(left[node])
        nodes_splits.append(right[node])
    return (nodes_splits,amount_of_branches)

def plot_tree(tree, feature_names):
    from matplotlib import gridspec
    import matplotlib.pyplot as plt
    from matplotlib import rc
    import pylab

    color = plt.cm.coolwarm(np.linspace(1,0,len(feature_names)))

    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')
    plt.rc('font', size=14)

    params = {'legend.fontsize': 20,
             'axes.labelsize': 20,
             'axes.titlesize':25,
             'xtick.labelsize':20,
             'ytick.labelsize':20}
    plt.rcParams.update(params)

    max_depth=tree.max_depth
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    fig = plt.figure(figsize=(3*2**max_depth,2*2**max_depth))
    gs = gridspec.GridSpec(max_depth, 2**max_depth)
    plt.subplots_adjust(hspace = 0.6, wspace=0.8)

    # All data
    amount_of_branches=1
    nodes=[0]
    normalize=np.sum(value[0][0])

    for i,node in enumerate(nodes):
        ax=fig.add_subplot(gs[0,(2**max_depth*i)/amount_of_branches:(2**max_depth*(i+1))/amount_of_branches])
        ax.set_title( features[node]+"$<= "+str(threshold[node])+"$")
        if( i==0): ax.set_ylabel(r'$\%$')
        ind=np.arange(1,len(value[node][0])+1,1)
        width=0.2
        bars= (np.array(value[node][0])/normalize)*100
        plt.bar(ind-width/2, bars, width,color=color,alpha=1,linewidth=0)
        plt.xticks(ind, [int(i) for i in ind-1])
        pylab.ticklabel_format(axis='y',style='sci',scilimits=(0,2))

    # Splits
    for j in range(1,max_depth):
        nodes,amount_of_branches=give_nodes(nodes,amount_of_branches,left,right)
        for i,node in enumerate(nodes):
            ax=fig.add_subplot(gs[j,(2**max_depth*i)/amount_of_branches:(2**max_depth*(i+1))/amount_of_branches])
            ax.set_title( features[node]+"$<= "+str(threshold[node])+"$")
            if( i==0): ax.set_ylabel(r'$\%$')
            ind=np.arange(1,len(value[node][0])+1,1)
            width=0.2
            bars= (np.array(value[node][0])/normalize)*100
            plt.bar(ind-width/2, bars, width,color=color,alpha=1,linewidth=0)
            plt.xticks(ind, [int(i) for i in ind-1])
            pylab.ticklabel_format(axis='y',style='sci',scilimits=(0,2))


    plt.tight_layout()
    return fig

# Example:

X=[]
Y=[]
amount_of_labels=5
feature_names=[ '$x_1$','$x_2$','$x_3$','$x_4$','$x_5$']
for i in range(200):
    X.append([np.random.normal(),np.random.randint(0,100),np.random.uniform(200,500) ])
    Y.append(np.random.randint(0,amount_of_labels))

clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=4)
clf = clf.fit(X,Y )
fig=plot_tree(clf, feature_names)

