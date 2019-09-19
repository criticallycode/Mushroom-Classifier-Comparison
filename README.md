# Mushroom-Classifier-Comparison
A comparison of different classifiers on a mushroom dataset.

This repo contains examples of visualizing data with Matplotlib and Seaborn, as well as implementing various classifiers on the Mushroom Dataset, available [here](https://www.kaggle.com/uciml/mushroom-classification).

The goal of this repo is to help instruct others on how to visualize and classify a dataset, and for that reason there is an attached IPython notebook that contains breakdowns of all the code blocks in the script.

This might go without saying, but don't take advice about which  mushrooms to eat from an IPython notebook. I do not condone eating any  mushrooms based on the patterns revealed in this notebook or in the  accompanying Python script or documentation.

Here is a brief outline of the script.

This section loads in the data and encodes it.

```Python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report, log_loss
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

m_data = pd.read_csv('mushrooms.csv')

# check for any null values
print(m_data.isnull().sum())

# see all unique values in a category
print(m_data['class'].unique())

# machine learning systems work with integers, we need to encode these
# string characters into ints

encoder = LabelEncoder()
# now apply the transformation to all the columns:
for col in m_data.columns:
    m_data[col] = encoder.fit_transform(m_data[col])

print(m_data.head(5))
```

This section visualizes some of the data.

```Python
# let's see how many poisonous and edible there are of each, 1 is poisonous, 0 is edible
# check to get a rough idea of correlations
correlations = m_data.corr()
plt.subplots(figsize=(20, 15))
#plt.figure(figsize=(16, 14))
data_corr = sns.heatmap(correlations, annot=True, linewidths=0.5, cmap="RdBu_r")
plt.show(data_corr)

# makes line plot of how various features are correlated with poisonous/edible class

features = ["cap-surface", "gill-attachment", "gill-size", "veil-color", "spore-print-color", "population", "habitat"]

for feature in features:
    line_plot = sns.lineplot(x="class", y=feature, label=feature, data=m_data)

plt.legend(loc=1)
plt.show()

# makes factor plot of how various gill colors are correlated with poisonous/edible class

gill_names =["buff","red","gray","chocolate","black","brown","orange","pink","green","purple","white","yellow"]
gill_colors = ["khaki","Red","darkGrey","chocolate","Black","saddleBrown","orange","lightpink","darkGreen","purple","lightGrey","Yellow"]

factor = sns.factorplot(x="gill-color",y="class",data=m_data, kind="bar", size = 8,
palette = gill_colors)
factor.set_xticklabels(rotation=45)
factor.set(xticks=range(0,14), xticklabels=gill_names)
factor = factor.set_ylabels("Prob. Poison")
plt.grid(axis='y')
plt.show()
```

This section separates the data into features and labels and scales the data.

```Python
X_features = m_data.iloc[:,1:23]  # all the features
y_label = m_data.iloc[:, 0]  # only the class/label
#let's confirm we're slicing this right
print(X_features.head())
print(y_label.head())
print(X_features.describe)

# we should probably scale the features, so that SVm or gaussian NB can deliver better predictions
scaler = StandardScaler()
X_features = scaler.fit_transform(X_features)
```

This next section carries out PCA on the dataset.

```Python
# principal component analysis, may or may not want to do, small dataset
pca = PCA()
pca.fit_transform(X_features)
plt.figure(figsize=(10,10))
# plot the explained variance ratio
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.grid()
plt.show()
# it looks like about 17 of the features account for about 95% of the variance in the dataset

# here's another way to visualize this
pca_variance = pca.explained_variance_

plt.figure(figsize=(8, 6))
plt.bar(range(22), pca_variance, alpha=0.5, align='center', label='individual variance')
plt.legend()
plt.ylabel('Variance ratio')
plt.xlabel('Principal components')
plt.show()

pca2 = PCA(n_components=17)
x_new = pca2.fit_transform(X_features)
X_train, X_test, y_train, y_test = train_test_split(x_new, y_label, test_size=0.20, random_state=2)
```

This next section instantiates and fits multiple classifiers, optimizing them with GridSearchCV and then saving their best parameters as a new classifier.

```Python
logreg_clf = LogisticRegression()
parameters_logreg = {"penalty": ["l2"], "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                     "max_iter": [25, 50, 100, 200, 400]}
grid_logreg = GridSearchCV(logreg_clf, parameters_logreg)
grid_logreg.fit(X_train, y_train)
logreg_opt = grid_logreg.best_estimator_

GNB_clf = GaussianNB()
GNB_clf.fit(X_train, y_train)

svc_clf = SVC()
svc_param = {"kernel": ["rbf", "linear"]}
grid_svc = GridSearchCV(svc_clf, svc_param)
grid_svc.fit(X_train, y_train)
svc_opt = grid_svc.best_estimator_

rf_clf = RandomForestClassifier()
parameters_rf = {"n_estimators": [4, 6, 8, 10, 12, 14, 16], "criterion": ["gini", "entropy"], "max_features": ["auto", "sqrt", "log2"],
                 "max_depth": [2, 3, 5, 10], "min_samples_split": [2, 3, 5, 10]}
grid_rf = GridSearchCV(rf_clf, parameters_rf)
grid_rf.fit(X_train, y_train)
rf_opt = grid_rf.best_estimator_

knn_clf = KNeighborsClassifier()
parameters_knn = {"n_neighbors": [3, 5, 10, 15, 20], "weights": ["uniform", "distance"],
                  "leaf_size": [10, 20, 30, 45, 60]}
grid_knn = GridSearchCV(knn_clf, parameters_knn)
grid_knn.fit(X_train, y_train)
knn_opt = grid_knn.best_estimator_

dt_clf = DecisionTreeClassifier()
parameters_dt = {"criterion": ["gini", "entropy"], "splitter": ["best", "random"], "max_features": ["auto", "log2", "sqrt"]}
grid_dt = GridSearchCV(dt_clf, parameters_dt)
grid_dt.fit(X_train, y_train)
dt_opt = grid_dt.best_estimator_

xgb_clf = XGBClassifier()
parameters_xg = {"objective" : ["reg:linear"], "n_estimators" : [5, 10, 15, 20]}
grid_xg = GridSearchCV(xgb_clf, parameters_xg)
grid_xg.fit(X_train, y_train)
xgb_opt = grid_xg.best_estimator_
```

Finally, this section carries our classification and returns metrics for the chosen classifiers.

```Python
chosen_metrics = [accuracy_score, log_loss, classification_report]

def get_reports(metrics, classifier):

    preds = classifier.predict(X_test)
    print("'{}' Performance: ".format(classifier.__class__.__name__))
    acc = accuracy_score(y_test, preds)
    l_loss = log_loss(y_test, preds)
    c_report = classification_report(y_test, preds)
    print("Accuracy: " + str(acc))
    print("Log Loss: " + str(l_loss))
    print("Classificaiton Report: ")
    print(c_report)
    print("----------")

get_reports(chosen_metrics, logreg_opt)
get_reports(chosen_metrics, GNB_clf)
get_reports(chosen_metrics, svc_opt)
get_reports(chosen_metrics, rf_opt)
get_reports(chosen_metrics, knn_opt)
get_reports(chosen_metrics, dt_opt)
get_reports(chosen_metrics, xgb_opt)
```

