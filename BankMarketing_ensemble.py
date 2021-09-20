import os
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import classification_report

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


#################################
# load data and preprocess it   #
#################################

# use "data_final"
source = "data_final"
# if ensemble is not used, upsample_input = 3 should be set for non-tree models -> higher model performance
#upsample_input = 3
upsample_input = 1
data = pd.read_csv(source+'.csv')


print('This Script is from', source)
print('it is upsampled times:', upsample_input)


data_numerical = pd.get_dummies(data, drop_first=True)
data_numerical = data_numerical.drop('y_yes', axis=1)

# creating X and Y categories
X_ori = data_numerical
Y = data['y']

# Normalize the input variables
X = (X_ori - X_ori.min()) / (X_ori.max() - X_ori.min())
#print(X.head(6)) # testing


from sklearn.model_selection import train_test_split, cross_val_score

# raised to train_size of 0.8
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# # upsampling not used in the current approach
# # (implement for non-ensemble approach with 3 differently transformed datasets for non-trees, trees and LightGBM
# # UPSAMPLING:
# from sklearn.utils import resample
# train_data = X_train.copy()
# train_data["y"] = Y_train
# train_data_minority = train_data.loc[train_data["y"] == "yes"]
# train_data_majority = train_data.loc[train_data["y"] == "no"]
# # for checking class sizes:
# print("majority: {}, minority: {}".format(len(train_data_majority), len(train_data_minority)))
# # change parameter 'n_samples' to change upsampled data set size
# train_data_minority_upsampled = resample(train_data_minority,
#                            replace=True,
#                            n_samples=len(train_data_minority)*upsample_input,
#                            random_state=0)
# train_data_upsampled = pd.concat([train_data_minority_upsampled, train_data_majority])
# X_train = train_data_upsampled.copy().iloc[:, :-1]
# Y_train = train_data_upsampled["y"]
# print("upsampled data set class counts: ", Y_train.value_counts())

# add the specific hyper-parameters, that lead to the highest result, to the report df
report = pd.DataFrame(columns=['Model', "Best Params", 'Acc. Train', 'Acc. Test', 'F1-Score Test', 'AUC', 'precision', 'recall'])


# ------------------------------------ #
# used for custom ensemble, DEPRECATED #
# ------------------------------------ #
# y_test_global = Y_test
# y_train_global = Y_train
# ensemble_y_train_list = []
# ensemble_y_pred_list = []
#
# X_train_index = X_train.index
# Y_train_index = Y_train.index
# X_test_index = X_test.index
# Y_test_index = Y_test.index
#
# print(X_train)
# print(Y_train)
# print(X_test)
# print(Y_test)

#################
#   Functions   #
#################


# function for creating a plot
def create_gridsearch_plot(model_gs, param, param_label, title):
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel(param_label)
    ax1.set_ylabel('Mean Accuracy', color=color)
    ax1.plot(model_gs.param_grid[param], model_gs.cv_results_["mean_test_score"], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Standard Deviation', color=color)  # we already handled the x-label with ax1
    ax2.plot(model_gs.param_grid[param], model_gs.cv_results_["std_test_score"], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Comparison of Accuracies and Standard Deviation ({})'.format(title))
    plt.xticks(model_gs.param_grid[param])
    plt.show()


# function to calculate cmtr, acctr, cmte, accte
def calculate_metrics(Y_train, Y_test, Y_train_pred, Y_test_pred):
    cmtr = confusion_matrix(Y_train, Y_train_pred)
    acctr = accuracy_score(Y_train, Y_train_pred)
    cmte = confusion_matrix(Y_test, Y_test_pred)
    accte = accuracy_score(Y_test, Y_test_pred)
    return {"cmtr": cmtr, "acctr": acctr, "cmte": cmte, "accte": accte}


# Visualize Confusion Matrix
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
def confusion_matrix_plotter(model, X_test, Y_test):
    plot_confusion_matrix(model, X_test, Y_test, labels=['no', 'yes'],
                          cmap=plt.cm.Blues, values_format='d')
    plt.show()


# function to caluclate precision metric
def calculate_precision_recall(Y_test, Y_test_pred):
    Y_test_numb = pd.get_dummies(Y_test, drop_first=True)
    Y_test_pred_numb = pd.get_dummies(Y_test_pred, drop_first=True)
    precision = round(metrics.precision_score(Y_test_numb, Y_test_pred_numb), 4)
    recall = round(metrics.recall_score(Y_test_numb, Y_test_pred_numb), 4)
    return {"precision": precision, "recall": recall}


# calculate f1 score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

# function to caluclate f1 score
def calculate_f1_score(Y_test, Y_test_pred):
    lb_churn = LabelEncoder()
    Y_test_code = lb_churn.fit_transform(Y_test)
    Y_test_pred_code = lb_churn.fit_transform(Y_test_pred)
    f1te = f1_score(Y_test_code, Y_test_pred_code)
    print(f1te)
    return f1te


# calculate ROC and AUC and plot the curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# function to calculate ROC and AUC and plot the curve
def calculate_roc_auc(model, X_test, Y_test):
    Y_probs = model.predict_proba(X_test)
    #print(Y_probs[0:6, :]) # testing
    Y_test_probs = np.array(np.where(Y_test == 'yes', 1, 0))
    #print(Y_test_probs[0:6]) # testing
    fpr, tpr, threshold = roc_curve(Y_test_probs, Y_probs[:, 1])
    #print(fpr, tpr, threshold) # testing
    roc_auc = auc(fpr, tpr)
    #print(roc_auc) # testing
    # Precision-recall-curve
    Y_test_probs = np.array(np.where(Y_test == 'yes', 1, 0))
    Y_test_pred_probs = np.array(np.where(Y_test_pred == 'yes', 1, 0))
    average_precision = average_precision_score(Y_test_probs, Y_test_pred_probs)
    disp = plot_precision_recall_curve(model, X_test, Y_test)
    disp.ax_.set_title('2-class Precision-Recall curve: '
                       'AP={0:0.2f}'.format(average_precision))
    plt.show()
    return {"roc_auc": roc_auc, "fpr": fpr, "tpr": tpr}


# plot model metrics fpr, tpr, roc_auc
import matplotlib.pyplot as plt

# function to plot model metrics fpr, tpr, roc_auc
def plot_model_metrics(metrics_dict, title):
    plt.plot(metrics_dict["fpr"], metrics_dict["tpr"], 'b', label='AUC = %0.2f' % metrics_dict["roc_auc"])
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title(title)
    plt.show()


# FOR TREES only
# show feature importance
def plot_feature_importance(model, title):
    list(zip(X, model.feature_importances_))
    index = np.arange(len(model.feature_importances_))
    bar_width = 1.0
    plt.bar(index, model.feature_importances_, bar_width)
    plt.xticks(index, list(X), rotation=90)  # labels get centered
    plt.title(title)
    plt.show()



from tabulate import tabulate
# formerly using tabulate for plotting, not implemented at the moment
# function to return information about the two param variations and their outcomes
def create_tabulate_plot(model_gs, param_1, param_2, param_label_1, param_label_2, title):
    print(pd.DataFrame({param_1: model_gs.cv_results_["param_{}".format(param_1)], param_2: model_gs.cv_results_["param_{}".format(param_2)], "mean_test_score": model_gs.cv_results_["mean_test_score"], "std_test_score": model_gs.cv_results_["std_test_score"]}))
    # not yet working, refactoring needed:
    print("Best accuracy is: ".format(np.max(model_gs.cv_results_["mean_test_score"])))
    maxi = model_gs.best_params_
    print("Best params are: {}".format(maxi))



#####################
#  START OF MODELS  #
#####################


################
#     KNN      #
################

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# new GridSearchCV implementation
# attention: the parameter 'n_jobs=-1' means all CPU cores will be used. This may result in an overflow of the RAM
model_gs_param_grid = {"n_neighbors": [6]}
model_gs = sk.model_selection.GridSearchCV(estimator=KNeighborsClassifier(), param_grid=model_gs_param_grid,
                                           scoring='accuracy', cv=10, n_jobs=-1)
model_gs.fit(X_train, Y_train)


# plot gridsearch
create_gridsearch_plot(model_gs, "n_neighbors", "Number of Neighbors", "k-NN")


# implement best model:
knnmodel = model_gs.best_estimator_
cv_score = cross_val_score(knnmodel, X_train, Y_train, cv=10, scoring="accuracy").mean()
knnmodel.fit(X_train, Y_train)
Y_train_pred = knnmodel.predict(X_train)
Y_test_pred = knnmodel.predict(X_test)


# calculate cmtr, acctr, cmte, accte
metrics_dict = calculate_metrics(Y_train, Y_test, Y_train_pred, Y_test_pred)

# override acctr by using cross-validation:
metrics_dict["acctr"] = cv_score


# plot confusion matrix
confusion_matrix_plotter(knnmodel, X_test, Y_test)


# plot classification report
print('Classification Report \n', classification_report(Y_test, Y_test_pred))


# calculate precision & recall metrics
metrics_dict.update(calculate_precision_recall(Y_test, Y_test_pred))


# calculate f1 score
metrics_dict["f1te"] = calculate_f1_score(Y_test, Y_test_pred)


# calculate metrics roc_auc, fpr, tpr
roc_auc_results = calculate_roc_auc(knnmodel, X_test, Y_test)
metrics_dict["roc_auc"] = roc_auc_results["roc_auc"]
metrics_dict["fpr"] = roc_auc_results["fpr"]
metrics_dict["tpr"] = roc_auc_results["tpr"]


# plot model metrics
plot_model_metrics(metrics_dict, 'ROC Curve of k-NN')


# add metrics to the report:
report.loc[len(report)] = ['k-NN', model_gs.best_params_, metrics_dict["acctr"], metrics_dict["accte"], metrics_dict["f1te"], metrics_dict["roc_auc"], metrics_dict["precision"], metrics_dict["recall"]]

# testing:
print(report)

results = pd.DataFrame(model_gs.cv_results_)
results.to_csv('Testing/'+source+'_KNN_report.csv', index = False)




###############
# Naive Bayes #
###############

from sklearn.naive_bayes import GaussianNB


# no grid search applied
nbmodel = GaussianNB()
cv_score = cross_val_score(nbmodel, X_train, Y_train, cv=10, scoring="accuracy").mean()
nbmodel.fit(X_train, Y_train)
Y_train_pred = nbmodel.predict(X_train)
Y_test_pred = nbmodel.predict(X_test)

# calculate cmtr, acctr, cmte, accte
metrics_dict = calculate_metrics(Y_train, Y_test, Y_train_pred, Y_test_pred)

# override acctr by using cross-validation:
metrics_dict["acctr"] = cv_score

# plot confusion matrix
confusion_matrix_plotter(nbmodel, X_test, Y_test)


# print classification report
print('Classification Report \n', classification_report(Y_test, Y_test_pred))


# calculate precision & recall metrics
metrics_dict.update(calculate_precision_recall(Y_test, Y_test_pred))


# calculate f1 score
metrics_dict["f1te"] = calculate_f1_score(Y_test, Y_test_pred)


# calculate ROC and AUC and plot the curve
roc_auc_results = calculate_roc_auc(nbmodel, X_test, Y_test)
metrics_dict["roc_auc"] = roc_auc_results["roc_auc"]
metrics_dict["fpr"] = roc_auc_results["fpr"]
metrics_dict["tpr"] = roc_auc_results["tpr"]


# plot the metrics
plot_model_metrics(metrics_dict, 'ROC Curve of GaussianNB')


# add metrics to the report:
report.loc[len(report)] = ['Naive Bayes', None, metrics_dict["acctr"], metrics_dict["accte"], metrics_dict["f1te"], metrics_dict["roc_auc"], metrics_dict["precision"], metrics_dict["recall"]]


# testing
print(report)



#########################
# Discriminant Analysis #
#########################

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# no grid search applied
dismodel = LinearDiscriminantAnalysis()
cv_score = cross_val_score(dismodel, X_train, Y_train, cv=10, scoring="accuracy").mean()
dismodel.fit(X_train, Y_train)
Y_train_pred = dismodel.predict(X_train)
Y_test_pred = dismodel.predict(X_test)

# calculate cmtr, acctr, cmte, accte
metrics_dict = calculate_metrics(Y_train, Y_test, Y_train_pred, Y_test_pred)

# override acctr by using cross-validation:
metrics_dict["acctr"] = cv_score

# Visualize Confusion Matrix
confusion_matrix_plotter(dismodel, X_test, Y_test)


# print classification report
print('Classification Report \n', classification_report(Y_test, Y_test_pred))

# calculate precision & recall metrics
metrics_dict.update(calculate_precision_recall(Y_test, Y_test_pred))


# calculate f1 score
metrics_dict["f1te"] = calculate_f1_score(Y_test, Y_test_pred)


# calculate ROC and AUC and plot the curve
roc_auc_results = calculate_roc_auc(dismodel, X_test, Y_test)
metrics_dict["roc_auc"] = roc_auc_results["roc_auc"]
metrics_dict["fpr"] = roc_auc_results["fpr"]
metrics_dict["tpr"] = roc_auc_results["tpr"]
plot_model_metrics(metrics_dict, 'ROC Curve of Linear Discriminate Analysis')


# add metrics to the report
report.loc[len(report)] = ['Linear Discriminant Analysis', None, metrics_dict["acctr"], metrics_dict["accte"], metrics_dict["f1te"], metrics_dict["roc_auc"], metrics_dict["precision"], metrics_dict["recall"]]


# testing
print(report)




#####################################
# Quadratic Disciminant Analysis    #
#####################################

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# no grid search applied
qdismodel = QuadraticDiscriminantAnalysis()
cv_score = cross_val_score(qdismodel, X_train, Y_train, cv=10, scoring="accuracy").mean()
qdismodel.fit(X_train, Y_train)
Y_train_pred = qdismodel.predict(X_train)
Y_test_pred = qdismodel.predict(X_test)

# calculate cmtr, acctr, cmte, accte
metrics_dict = calculate_metrics(Y_train, Y_test, Y_train_pred, Y_test_pred)

# override acctr by using cross-validation:
metrics_dict["acctr"] = cv_score

# Visualize Confusion Matrix
confusion_matrix_plotter(qdismodel, X_test, Y_test)


# print classification report
print('Classification Report \n', classification_report(Y_test, Y_test_pred))


# calculate precision & recall metrics
metrics_dict.update(calculate_precision_recall(Y_test, Y_test_pred))


# calculate f1 score
metrics_dict["f1te"] = calculate_f1_score(Y_test, Y_test_pred)


# calculate ROC and AUC and plot the curve
roc_auc_results = calculate_roc_auc(qdismodel, X_test, Y_test)
metrics_dict["roc_auc"] = roc_auc_results["roc_auc"]
metrics_dict["fpr"] = roc_auc_results["fpr"]
metrics_dict["tpr"] = roc_auc_results["tpr"]
plot_model_metrics(metrics_dict, 'ROC Curve of Quadratic Discriminate Analysis')


# add metrics to the report
report.loc[len(report)] = ['Quadratic Discriminant Analysis', None, metrics_dict["acctr"], metrics_dict["accte"], metrics_dict["f1te"], metrics_dict["roc_auc"], metrics_dict["precision"], metrics_dict["recall"]]


# testing
print(report)



#######################
# Logistic Regression #
#######################

from sklearn.linear_model import LogisticRegression


# no grid search applied
lrmodel = LogisticRegression(class_weight="balance", random_state=0)
cv_score = cross_val_score(lrmodel, X_train, Y_train, cv=10, scoring="accuracy").mean()
lrmodel.fit(X_train, Y_train)
Y_train_pred = lrmodel.predict(X_train)
Y_test_pred = lrmodel.predict(X_test)


# calculate cmtr, acctr, cmte, accte
metrics_dict = calculate_metrics(Y_train, Y_test, Y_train_pred, Y_test_pred)

# override acctr by using cross-validation:
metrics_dict["acctr"] = cv_score

# Visualize Confusion Matrix
confusion_matrix_plotter(lrmodel, X_test, Y_test)


# print classification report:
print('Classification Report \n', classification_report(Y_test, Y_test_pred))


# calculate precision & recall metrics
metrics_dict.update(calculate_precision_recall(Y_test, Y_test_pred))


# calculate f1 score
metrics_dict["f1te"] = calculate_f1_score(Y_test, Y_test_pred)


# calculate ROC and AUC and plot the curve
roc_auc_results = calculate_roc_auc(lrmodel, X_test, Y_test)
metrics_dict["roc_auc"] = roc_auc_results["roc_auc"]
metrics_dict["fpr"] = roc_auc_results["fpr"]
metrics_dict["tpr"] = roc_auc_results["tpr"]
plot_model_metrics(metrics_dict, 'ROC Curve of Logistic Regression')


report.loc[len(report)] = ['Logistic Regression', None, metrics_dict["acctr"], metrics_dict["accte"], metrics_dict["f1te"], metrics_dict["roc_auc"], metrics_dict["precision"], metrics_dict["recall"]]

# testing
print(report)




##################
# Neural Network #
##################

from sklearn.neural_network import MLPClassifier


model_gs_param_grid = {"hidden_layer_sizes": [(10,10)], 'max_iter': [500]}
model_gs = sk.model_selection.GridSearchCV(estimator=MLPClassifier(random_state=0, activation='relu'), param_grid=model_gs_param_grid,
                                           scoring='accuracy', cv=10, n_jobs=-1)
model_gs.fit(X_train, Y_train)
print(model_gs.cv_results_)

# save cross-validated score of gridsearch
cv_score = model_gs.best_score_

# creating a Plot
create_tabulate_plot(model_gs, "hidden_layer_sizes", "max_iter", "hidden layer size of the NN", "maximum Iterations", "Neural Network")


# create model using the optimal parameters
nnetmodel = model_gs.best_estimator_
nnetmodel.fit(X_train, Y_train)
Y_train_pred = nnetmodel.predict(X_train)
Y_test_pred = nnetmodel.predict(X_test)

# calculate cmtr, acctr, cmte, accte
metrics_dict = calculate_metrics(Y_train, Y_test, Y_train_pred, Y_test_pred)

# override acctr by using cross-validation:
metrics_dict["acctr"] = cv_score

# plot confusion matrix
confusion_matrix_plotter(nnetmodel, X_test, Y_test)


# print classification report
print('Classification Report \n', classification_report(Y_test, Y_test_pred))


# calculate precision & recall metrics
metrics_dict.update(calculate_precision_recall(Y_test, Y_test_pred))


# calculate f1 score
metrics_dict["f1te"] = calculate_f1_score(Y_test, Y_test_pred)


# calculate ROC and AUC and plot the curve
roc_auc_results = calculate_roc_auc(nnetmodel, X_test, Y_test)
metrics_dict["roc_auc"] = roc_auc_results["roc_auc"]
metrics_dict["fpr"] = roc_auc_results["fpr"]
metrics_dict["tpr"] = roc_auc_results["tpr"]
plot_model_metrics(metrics_dict, 'ROC Curve of Neural Network')


# add metrics to the report
report.loc[len(report)] = ['Neural Network', model_gs.best_params_, metrics_dict["acctr"], metrics_dict["accte"], metrics_dict["f1te"], metrics_dict["roc_auc"], metrics_dict["precision"], metrics_dict["recall"]]


# testing
print(report)

results = pd.DataFrame(model_gs.cv_results_)
results.to_csv('Testing/'+source+'_NN_report.csv', index = False)


#############
#   SVC     #
#############

from sklearn.svm import LinearSVC

model_gs_param_grid = {"C": [13]}
model_gs = sk.model_selection.GridSearchCV(estimator=LinearSVC(random_state=0, max_iter= 3000, class_weight="balanced"), param_grid=model_gs_param_grid,
                                           scoring='accuracy', cv=10, n_jobs=-1)

model_gs.fit(X_train, Y_train)
print(model_gs.cv_results_)

# save cross-validated score of gridsearch
cv_score = model_gs.best_score_

# creating a Plot
create_gridsearch_plot(model_gs, "C", "C of SVC", "Linear SVC")


# create model using the optimal parameters
svcmodel = model_gs.best_estimator_
svcmodel.fit(X_train, Y_train)
Y_train_pred = svcmodel.predict(X_train)
Y_test_pred = svcmodel.predict(X_test)


# calculate cmtr, acctr, cmte, accte
metrics_dict = calculate_metrics(Y_train, Y_test, Y_train_pred, Y_test_pred)

# override acctr by using cross-validation:
metrics_dict["acctr"] = cv_score

# Visualize Confusion Matrix
confusion_matrix_plotter(svcmodel, X_test, Y_test)


# plot classification report
print('Classification Report SVC: \n', classification_report(Y_test, Y_test_pred))


# calculate precision & recall metrics
metrics_dict.update(calculate_precision_recall(Y_test, Y_test_pred))


# calculate f1 score
metrics_dict["f1te"] = calculate_f1_score(Y_test, Y_test_pred)


metrics_dict["roc_auc"] = "Not available"

# # calculate ROC and AUC and plot the curve -> not available with LinearSVC
# # we could use SVC(probability=True), but this is significantly slower and the results are not good enough to use in
# # our final ensemble. So we will not plot roc_auc
# roc_auc_results = calculate_roc_auc(svcmodel, X_test, Y_test)
# metrics_dict["roc_auc"] = roc_auc_results["roc_auc"]
# metrics_dict["fpr"] = roc_auc_results["fpr"]
# metrics_dict["tpr"] = roc_auc_results["tpr"]
# plot_model_metrics(metrics_dict, 'ROC Curve of SVC')


# add metrics to the report
report.loc[len(report)] = ['SVC', model_gs.best_params_, metrics_dict["acctr"], metrics_dict["accte"], metrics_dict["f1te"], metrics_dict["roc_auc"], metrics_dict["precision"], metrics_dict["recall"]]


results = pd.DataFrame(model_gs.cv_results_)
results.to_csv('Testing/'+source+'_SVC_report.csv', index = False)



#########################################################
#              Tree classifiers                         #
#-> preprocess data differently (no upsampling)         #
#--> currently not in use due to ensembling approach    #
#########################################################


# # use "data_final"
# source = "data_final"
# upsample_input = 1
# data = pd.read_csv(source+'.csv')
#
#
# print('This Script is from', source)
# print('it is upsampeld times:', upsample_input)
#
#
# data_numerical = pd.get_dummies(data, drop_first=True)
# data_numerical = data_numerical.drop('y_yes', axis=1)
#
# # creating X and Y categories
# X_ori = data_numerical
# Y = data['y']
#
# # Normalize the input variables
# X = (X_ori - X_ori.min()) / (X_ori.max() - X_ori.min())
# #print(X.head(6)) # testing
#
# # Normalize the input variables
# X = (X_ori - X_ori.min()) / (X_ori.max() - X_ori.min())
# #print(X.head(6)) # testing


# approach for custom upsampling - DEPRECATED due to new approach: select data_final without upsampling for all models
# different approach due to ensembling with different datasets: select rows by indices of first train_test_split

# from sklearn.model_selection import train_test_split
#
# # raised to train size of 0.8
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#
#
# print("Y_test is equals to Y_test_global: ", Y_test.equals(y_test_global))


# select same rows as in initial train_test_split
# X_train = X_ori[X_ori.index.isin(X_train_index)]
# Y_train = Y[Y.index.isin(Y_train_index)]
# X_test = X_ori[X_ori.index.isin(X_test_index)]
# Y_test = Y[Y.index.isin(Y_test_index)]


######################
#   Decision Trees   #
######################

from sklearn.tree import DecisionTreeClassifier


model_gs_param_grid = {'max_depth': [4]}
model_gs = sk.model_selection.GridSearchCV(estimator=DecisionTreeClassifier(criterion='entropy', random_state=0, class_weight='balanced'), param_grid=model_gs_param_grid,
                                           scoring='accuracy', cv=10, n_jobs=-1)
model_gs.fit(X_train, Y_train)
print(model_gs.cv_results_)

# save cross-validated score of gridsearch
cv_score = model_gs.best_score_


# creating a Plot
create_gridsearch_plot(model_gs, "max_depth", "max depth of the tree", "Decision Tree")


# create model with optimal parameters
etmodel = model_gs.best_estimator_
etmodel.fit(X_train, Y_train)
Y_train_pred = etmodel.predict(X_train)
Y_test_pred = etmodel.predict(X_test)


# calculate cmtr, acctr, cmte, accte
metrics_dict = calculate_metrics(Y_train, Y_test, Y_train_pred, Y_test_pred)

# override acctr by using cross-validation:
metrics_dict["acctr"] = cv_score

# Visualize Confusion Matrix
confusion_matrix_plotter(etmodel, X_test, Y_test)


# plot classification report
print('Classification Report \n', classification_report(Y_test, Y_test_pred))


# calculate precision & recall metrics
metrics_dict.update(calculate_precision_recall(Y_test, Y_test_pred))


# calculate f1 score
metrics_dict["f1te"] = calculate_f1_score(Y_test, Y_test_pred)


# calculate ROC and AUC and plot the curve
roc_auc_results = calculate_roc_auc(etmodel, X_test, Y_test)
metrics_dict["roc_auc"] = roc_auc_results["roc_auc"]
metrics_dict["fpr"] = roc_auc_results["fpr"]
metrics_dict["tpr"] = roc_auc_results["tpr"]
plot_model_metrics(metrics_dict, 'ROC Curve of Decision Tree')


# plot feature importance
plot_feature_importance(etmodel, 'Decision Tree - Feature Importance')

# add metrics to the report
report.loc[len(report)] = ['Decision Tree', model_gs.best_params_, metrics_dict["acctr"], metrics_dict["accte"], metrics_dict["f1te"], metrics_dict["roc_auc"], metrics_dict["precision"], metrics_dict["recall"]]


# testing
print(report)

results = pd.DataFrame(model_gs.cv_results_)
results.to_csv('Testing/'+source+'_decision_tree_report.csv', index = False)


# # if wanted: display the tree as an image using graphviz
# #=============================================================================
# #show tree using graphviz
# import graphviz
# dot_data = sk.tree.export_graphviz(etmodel, out_file=None,
#                           feature_names=list(X),
#                           filled=True, rounded=True,
#                           special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.format = 'png'
# graph.render("Churn_entropy")
# #=============================================================================

#################
#     Gini      #
#################


model_gs_param_grid = {"max_depth": [4]}
model_gs = sk.model_selection.GridSearchCV(estimator=DecisionTreeClassifier(random_state=0, class_weight='balanced'), param_grid=model_gs_param_grid,
                                           scoring='accuracy', cv=10, n_jobs=-1)
model_gs.fit(X_train, Y_train)
print(model_gs.cv_results_)

# save cross-validated score of gridsearch
cv_score = model_gs.best_score_

results = pd.DataFrame(model_gs.cv_results_)
results.to_csv('Testing/'+source+'_gini_report.csv', index = False)

# creating a Plot
create_gridsearch_plot(model_gs, "max_depth", "max depth of the tree", "Gini Decision Tree")


# create model using the optimal parameters
gtmodel = model_gs.best_estimator_
gtmodel.fit(X_train, Y_train)
Y_train_pred = gtmodel.predict(X_train)
Y_test_pred = gtmodel.predict(X_test)

# calculate cmtr, acctr, cmte, accte
metrics_dict = calculate_metrics(Y_train, Y_test, Y_train_pred, Y_test_pred)

# override acctr by using cross-validation:
metrics_dict["acctr"] = cv_score

# Visualize Confusion Matrix
confusion_matrix_plotter(gtmodel, X_test, Y_test)


# print classification report
print('Classification Report \n', classification_report(Y_test, Y_test_pred))


# calculate precision & recall metrics
metrics_dict.update(calculate_precision_recall(Y_test, Y_test_pred))


# calculate f1 score
metrics_dict["f1te"] = calculate_f1_score(Y_test, Y_test_pred)


# calculate ROC and AUC and plot the curve
roc_auc_results = calculate_roc_auc(gtmodel, X_test, Y_test)
metrics_dict["roc_auc"] = roc_auc_results["roc_auc"]
metrics_dict["fpr"] = roc_auc_results["fpr"]
metrics_dict["tpr"] = roc_auc_results["tpr"]
plot_model_metrics(metrics_dict, 'ROC Curve of Gini Decision Tree')


# plot feature importance
plot_feature_importance(gtmodel, 'Gini Decision Tree - Feature Importance')

# add metrics to the report
report.loc[len(report)] = ['Gini', model_gs.best_params_, metrics_dict["acctr"], metrics_dict["accte"], metrics_dict["f1te"], metrics_dict["roc_auc"], metrics_dict["precision"], metrics_dict["recall"]]


# testing
print(report)


results = pd.DataFrame(model_gs.cv_results_)
results.to_csv('Testing/'+source+'_gini_report.csv', index = False)

# # part of custom ensemble, DEPRECATED
# ensemble_y_train_list.append(Y_train_pred)
# ensemble_y_pred_list.append(Y_test_pred)


# # if wanted: display the tree as an image using graphviz
# #=============================================================================
# #show tree using graphviz
# import graphviz
# dot_data = sk.tree.export_graphviz(gtmodel, out_file=None,
#                           feature_names=list(X),
#                           filled=True, rounded=True,
#                           special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.format = 'png'
# graph.render("Gini_Tree_Image")
# #=============================================================================


#################
# Random Forest #
#################

from sklearn.ensemble import RandomForestClassifier


model_gs_param_grid = {'max_depth': [10], 'n_estimators': [400]}
model_gs = sk.model_selection.GridSearchCV(estimator=RandomForestClassifier(random_state=0, class_weight='balanced'), param_grid=model_gs_param_grid,
                                           scoring='accuracy', cv=10, n_jobs=-1)
model_gs.fit(X_train, Y_train)
print('Random-Forest-Results-Training: /n:', model_gs.cv_results_)

# save cross-validated score of gridsearch
cv_score = model_gs.best_score_

# print GridSearchCV results
create_tabulate_plot(model_gs, "max_depth", "n_estimators", "max depth", "number of trees", "Random Forest")

# plot gridsearch
#create_gridsearch_plot(model_gs, "max_depth", "Maximum depth of the tree", "Random Forest")


# create model using the optimal parameters
rfmodel = model_gs.best_estimator_
rfmodel.fit(X_train, Y_train)
Y_train_pred = rfmodel.predict(X_train)
Y_test_pred = rfmodel.predict(X_test)

# calculate cmtr, acctr, cmte, accte
metrics_dict = calculate_metrics(Y_train, Y_test, Y_train_pred, Y_test_pred)

# override acctr by using cross-validation:
metrics_dict["acctr"] = cv_score

# Visualize Confusion Matrix
confusion_matrix_plotter(rfmodel, X_test, Y_test)


# print classification report
print('Classification Report \n', classification_report(Y_test, Y_test_pred))


# calculate precision & recall metrics
metrics_dict.update(calculate_precision_recall(Y_test, Y_test_pred))


# calculate f1 score
metrics_dict["f1te"] = calculate_f1_score(Y_test, Y_test_pred)


# calculate ROC and AUC and plot the curve
roc_auc_results = calculate_roc_auc(rfmodel, X_test, Y_test)
metrics_dict["roc_auc"] = roc_auc_results["roc_auc"]
metrics_dict["fpr"] = roc_auc_results["fpr"]
metrics_dict["tpr"] = roc_auc_results["tpr"]
plot_model_metrics(metrics_dict, 'ROC Curve of Random Forest')


# show feature importance
plot_feature_importance(rfmodel, 'Random Forest - Feature Importance')

results = pd.DataFrame(model_gs.cv_results_)
results.to_csv('Testing/'+source+'_random_forest_report.csv', index = False)

# add metrics to report
report.loc[len(report)] = ['Random-Forest', model_gs.best_params_, metrics_dict["acctr"], metrics_dict["accte"], metrics_dict["f1te"], metrics_dict["roc_auc"], metrics_dict["precision"], metrics_dict["recall"]]

# View a list of the features and their importance scores
print(list(zip(X_train, rfmodel.feature_importances_)))

# # part of custom ensemble, DEPRECATED
# ensemble_y_train_list.append(Y_train_pred)
# ensemble_y_pred_list.append(Y_test_pred)



#################################
# NEEDS TO BE THE LAST MODEL!!  # -> if approach with 3 different datasets is used for non-trees, trees and LightGBM
#################################
# LightGBM  #
#############

#
# # USES THE ORIGINAL DATASET WITHOUT DURATION
# source = "BankMarketing"
#
# data = pd.read_csv(source+'.csv')
#
# data = data.drop(columns='duration')
# data[data.select_dtypes('object').columns.tolist()] = data[data.select_dtypes('object').columns.tolist()].astype('category')
#
#
# # creating X and Y categories
# X = data.copy().drop(columns='y')
# Y = data['y']
#
# # # different approach due to ensembling with different datasets: select rows by indices of first train_test_split
# # from sklearn.model_selection import train_test_split
# #
# # # raised to 0.8
# # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# #
#
# # select same rows as initial train_test_split
# X_train = X[X.index.isin(X_train_index)]
# Y_train = Y[Y.index.isin(Y_train_index)]
# X_test = X[X.index.isin(X_test_index)]
# Y_test = Y[Y.index.isin(Y_test_index)]
#


import lightgbm as lgb

cat_col = X_train.select_dtypes('object').columns.tolist() + X_train.select_dtypes('category').columns.tolist()


d_train = lgb.Dataset(X_train, label=Y_train)
d_valid = lgb.Dataset(X_test, label=Y_test)
# for further exploration:
#d_train = lgb.Dataset(X_train, label=Y_train, feature_name=X_train.columns.tolist(), categorical_feature=cat_col)

print(X_train.columns.tolist())

model_gs_param_grid = {"max_depth": [5], "learning_rate": [0.005], "num_leaves": [31]}

# parameter for lgb.LGBMClassifier(feature_name=X_train.columns.tolist())
model_gs = sk.model_selection.GridSearchCV(estimator=lgb.LGBMClassifier(random_state=0, objective='binary', class_weight='balanced'), param_grid=model_gs_param_grid,
                                           scoring='accuracy', cv=10, n_jobs=-1)
model_gs.fit(X_train, Y_train)

# save cross-validated score of gridsearch
cv_score = model_gs.best_score_

# create model using the optimal parameters
lgbmodel = model_gs.best_estimator_
lgbmodel.fit(X_train, Y_train)
Y_train_pred = lgbmodel.predict(X_train)
Y_test_pred = lgbmodel.predict(X_test)


# calculate cmtr, acctr, cmte, accte
metrics_dict = calculate_metrics(Y_train, Y_test, Y_train_pred, Y_test_pred)

# override acctr by using cross-validation:
metrics_dict["acctr"] = cv_score

# Visualize Confusion Matrix
confusion_matrix_plotter(lgbmodel, X_test, Y_test)


# print classification report
print('Classification Report \n', classification_report(Y_test, Y_test_pred))


# calculate precision & recall metrics
metrics_dict.update(calculate_precision_recall(Y_test, Y_test_pred))


# calculate f1 score
metrics_dict["f1te"] = calculate_f1_score(Y_test, Y_test_pred)


# calculate ROC and AUC and plot the curve
roc_auc_results = calculate_roc_auc(lgbmodel, X_test, Y_test)
metrics_dict["roc_auc"] = roc_auc_results["roc_auc"]
metrics_dict["fpr"] = roc_auc_results["fpr"]
metrics_dict["tpr"] = roc_auc_results["tpr"]
plot_model_metrics(metrics_dict, 'ROC Curve of LightGBM')


# show feature importance
plot_feature_importance(lgbmodel, 'LightGBM - Feature Importance')

report.loc[len(report)] = ['LightGBM Classifier', model_gs.best_params_, metrics_dict["acctr"], metrics_dict["accte"], metrics_dict["f1te"], metrics_dict["roc_auc"], metrics_dict["precision"], metrics_dict["recall"]]


results = pd.DataFrame(model_gs.cv_results_)
results.to_csv('Testing/'+source+'_lightGBM_report.csv', index = False)


# # part of custom ensemble, DEPRECATED
# ensemble_y_train_list.append(Y_train_pred)
# ensemble_y_pred_list.append(Y_test_pred)



#############################
#           Ensemble        #
#############################

from mlxtend.classifier import EnsembleVoteClassifier

# use Gini Decision Tree, Random Forest, LightGBM
list_classifiers = [gtmodel, rfmodel, lgbmodel]

# hard voting because soft voting did not improve results
ens_model = EnsembleVoteClassifier(clfs=list_classifiers)

cv_score = cross_val_score(lrmodel, X_train, Y_train, cv=10, scoring="accuracy").mean()

ens_model.fit(X_train,Y_train)

Y_train_pred = ens_model.predict(X_train)
Y_test_pred = ens_model.predict(X_test)


# calculate cmtr, acctr, cmte, accte
metrics_dict = calculate_metrics(Y_train, Y_test, Y_train_pred, Y_test_pred)

# override acctr by using cross-validation:
metrics_dict["acctr"] = cv_score

# Visualize Confusion Matrix
confusion_matrix_plotter(ens_model, X_test, Y_test)

# print classification report
print('Classification Report \n', classification_report(Y_test, Y_test_pred))


# calculate precision & recall metrics
metrics_dict.update(calculate_precision_recall(Y_test, Y_test_pred))


# calculate f1 score
metrics_dict["f1te"] = calculate_f1_score(Y_test, Y_test_pred)


# calculate ROC and AUC and plot the curve
roc_auc_results = calculate_roc_auc(ens_model, X_test, Y_test)
metrics_dict["roc_auc"] = roc_auc_results["roc_auc"]
metrics_dict["fpr"] = roc_auc_results["fpr"]
metrics_dict["tpr"] = roc_auc_results["tpr"]
plot_model_metrics(metrics_dict, 'ROC Curve of Ensemble')

# add to the report
report.loc[len(report)] = ['Ensemble', None, metrics_dict["acctr"], metrics_dict["accte"], metrics_dict["f1te"], metrics_dict["roc_auc"], metrics_dict["precision"], metrics_dict["recall"]]

print(report)



# NOT IN USE ANYMORE, approach to create a CUSTOM ENSEMBLE with different datasets for the different models
# print("Y_test is equals to Y_test_global: ", Y_test.equals(y_test_global))
#
# (for optimized results per model)
# ensemble results for: Gini Decision Tree, Random Forest, LightGBM
# def get_ensemble_result(list_):
#     print(list_)
#     for j in list_:
#         print(j)
#     y_predictions = pd.DataFrame({0:list_[0], 1:list_[1], 2:list_[2]})
#     no_of_cols = len(y_predictions.columns)
#     threshold = no_of_cols/2
#     Y_ensemble_results = y_predictions.apply(lambda x: "yes" if ((x.value_counts().yes > 2) if hasattr(x.value_counts(), 'yes') else (x.value_counts().no<2)) else "no", axis=1)
#     print(Y_ensemble_results)
#     return Y_ensemble_results
#
# Y_train_pred_ensemble = get_ensemble_result(ensemble_y_train_list)
# Y_test_pred_ensemble = get_ensemble_result(ensemble_y_pred_list)

# calculate cmtr, acctr, cmte, accte
# metrics_dict = calculate_metrics(y_train_global, y_test_global, Y_train_pred_ensemble, Y_test_pred_ensemble)
#
#
#
# # print classification report
# print('Classification Report \n', classification_report(y_test_global, Y_test_pred_ensemble))
#
#
# # calculate precision & recall metrics
# metrics_dict.update(calculate_precision_recall(y_test_global, Y_test_pred_ensemble))
#
#
# # calculate f1 score
# metrics_dict["f1te"] = calculate_f1_score(y_test_global, Y_test_pred_ensemble)


#############################
# PRINT THE FINAL REPORT    #
#############################

#report.to_csv('Testing/report_all_'+source+'.csv', index=False)
report.to_csv('Testing/report_all_final_data.csv', index=False)

print("stop")



"""

#############################
#           Lime            #
#############################

import lime
import lime.lime_tabular

# look for values that 
print(Y_test.head(30))

predict_fn = lambda x: rfmodel.predict_proba(x).astype(float)
X = X_train.values
explainer = lime.lime_tabular.LimeTabularExplainer(X,feature_names= X_train.columns,class_names=['no', 'yes'],kernel_width=5)

chosen_customer = X_test.loc[[7672]].values[0]
explanation = explainer.explain_instance(chosen_customer,predict_fn,num_features=10)
fig1 = explanation.as_pyplot_figure();

fig1.tight_layout()
plot.savefig('lime_fig1.png', dpi=300)

# for notebook output
explanation.show_in_notebook(show_all=False)


chosen_customer_2 = X_test.loc[[28335]].values[0]
explanation = explainer.explain_instance(chosen_customer_2,predict_fn,num_features=10)
fig2 = explanation.as_pyplot_figure();

fig2.tight_layout()
plot.savefig('lime_fig2.png', dpi=300)

# for notebook output
explanation.show_in_notebook(show_all=False)

"""

"""
#############################
#           Shap            #
#############################
import shap
# for trees
shap_explainer = shap.TreeExplainer(ens_model) #again replace model if needed
# for non-trees -> NOTE: does not work for (voting) ensembles (not implemented in SHAP yet)
#shap_explainer = shap.KernelExplainer(ens_model, X_train) #again replace model if needed
test_shap_vals = shap_explainer.shap_values(X_test)
# test_shap_vals index 0 is impact on outcome="yes", 1 is impact on outcome="no"
shap.summary_plot(test_shap_vals[0], X_test)

"""
