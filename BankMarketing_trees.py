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
#   CHANGE your path here       #
#################################


# use the data from the folder 'data_different_variants' - which is the updated data
source = "data_all_1"
upsample_input = 1
data = pd.read_csv(source+'.csv')

# JUST FOR TESTING
#data = data.sample(frac=0.005)

print('This Script is from', source)
print('it is upsampeld times:', upsample_input)

# IMPLEMENTING DOWNSAMPLING !!BEFORE SPLITTING!! FOR TESTING PURPOSES:
# Separate majority and minority classes
#from sklearn.utils import resample
#
#df_majority = data[data.y == 'no']
#df_minority = data[data.y == 'yes']
#
#
#df_majority_downsampled = resample(df_majority,
#                                    replace=False,  # sample with replacement
#                                    n_samples=len(df_minority),  # to match majority class
#                                    random_state=0)  # reproducible results

# Combine minority class with downsampled majority class
#df_downsampled = pd.concat([df_minority, df_majority_downsampled])
#data = df_downsampled


data_numerical = pd.get_dummies(data, drop_first=True)
data_numerical = data_numerical.drop('y_yes', axis=1)

# creating X and Y categories
X_ori = data_numerical
Y = data['y']

# Normalize the input variables
X = (X_ori - X_ori.min()) / (X_ori.max() - X_ori.min())
print(X.head(6))

# Normalize the input variables
X = (X_ori - X_ori.min()) / (X_ori.max() - X_ori.min())
print(X.head(6))

from sklearn.model_selection import train_test_split

# raised to 0.7, maybe 0.8 later
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(Y_train.value_counts())
print(Y_test.value_counts())

print(X_train)
print(Y_train)

# no upsampling

# add the specific hyper-parameters, that lead to the highest result, to the report df
report = pd.DataFrame(columns=['Model', "Best Params", 'Acc. Train', 'Acc. Test', 'F1-Score Test', 'ROC', 'precision', 'recall'])


#################
#   Functions   #
#################
# INTO PRESENTATION
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


# INTO PRESENTATION
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
    # prints for verification:
    #print("y_test original: ", Y_test)
    #print("y_test_pred original: ", Y_test_pred)
    Y_test_code = lb_churn.fit_transform(Y_test)
    Y_test_pred_code = lb_churn.fit_transform(Y_test_pred)
    #print("y_test transformed before F1 score: ", Y_test_code)
    #print("y_pred transformed before F1 score: ", Y_test_pred_code)
    f1te = f1_score(Y_test_code, Y_test_pred_code)
    print(f1te)
    return f1te


# INTO PRESENTATION
# calculate ROC and AUC and plot the curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# functoin to calculate ROC and AUC and plot the curve
def calculate_roc_auc(model, X_test, Y_test):
    Y_probs = model.predict_proba(X_test)
    print(Y_probs[0:6, :])
    Y_test_probs = np.array(np.where(Y_test == 'yes', 1, 0))
    print(Y_test_probs[0:6])

    fpr, tpr, threshold = roc_curve(Y_test_probs, Y_probs[:, 1])
    print(fpr, tpr, threshold)

    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    # Precsion-recall-curve
    Y_test_probs = np.array(np.where(Y_test == 'yes', 1, 0))
    Y_test_pred_probs = np.array(np.where(Y_test_pred == 'yes', 1, 0))
    average_precision = average_precision_score(Y_test_probs, Y_test_pred_probs)
    disp = plot_precision_recall_curve(model, X_test, Y_test)
    disp.ax_.set_title('2-class Precision-Recall curve: '
                       'AP={0:0.2f}'.format(average_precision))
    plt.show()
    return {"roc_auc": roc_auc, "fpr": fpr, "tpr": tpr}


# plot metrics
import matplotlib.pyplot as plt
# INTO PRESENTATION
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


# INTO PRESENTATION
# FOR DECISION TREE
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
# STILL NEEDS SOME REFACTORING
# function to return information about the two param variations and their outcomes
def create_tabulate_plot(model_gs, param_1, param_2, param_label_1, param_label_2, title):
    #headers = [param_label_1, param_label_2, "Mean Accuracy", "Standard Deviation"]
    #table = tabulate(pd.DataFrame(model_gs.param_grid).transpose(), headers, tablefmt="plain", floatfmt=".3f")
    #print("\n", table)
    print(pd.DataFrame({param_1: model_gs.cv_results_["param_{}".format(param_1)], param_2: model_gs.cv_results_["param_{}".format(param_2)], "mean_test_score": model_gs.cv_results_["mean_test_score"], "std_test_score": model_gs.cv_results_["std_test_score"]}))
    # not yet working:
    print("Best accuracy is: ".format(np.max(model_gs.cv_results_["mean_test_score"])))
    maxi = model_gs.best_params_
    print("Best params are: {}".format(maxi))
    #table = tabulate(train_accuracies[:, maxi[1, :]].transpose(), headers, tablefmt="plain", floatfmt=".3f")
    #print('The best Acc is: ', "\n", table)

# CUT OUT BY REFACTORING:
#from tabulate import tabulate
#headers = ["Max_Depth", "n_Estimators", "Mean Accurancie", "Standard Deviation"]
#table = tabulate(train_accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
#print("\n", table)
#print(train_accuracies[2].max())
#maxi = np.array(np.where(train_accuracies == train_accuracies[2].max()))
#print(maxi[0, :], maxi[1, :])
#print(train_accuracies[:, maxi[1, :]])
#table = tabulate(train_accuracies[:, maxi[1, :]].transpose(), headers, tablefmt="plain", floatfmt=".3f")
#print('The best Acc is: ', "\n", table)

#
# ######################
# #   Decision Trees   #
# ######################
#
# from sklearn.tree import DecisionTreeClassifier
#
#
# # test parameter max_depth
# # new GridSearchCV implementation
# # attention: the parameter 'n_jobs=-1' means all CPU cores will be used. This may result in an overflow of the RAM
# # do not forget to change parameter 'cv'
# model_gs_param_grid = {'max_depth': [2, 3, 4, 5, 6]}
# model_gs = sk.model_selection.GridSearchCV(estimator=DecisionTreeClassifier(criterion='entropy', random_state=0, class_weight='balanced'), param_grid=model_gs_param_grid,
#                                            scoring='accuracy', cv=10, n_jobs=-1)
# model_gs.fit(X_train, Y_train)
# print(model_gs.cv_results_)
#
#
# # creating a Plot
# create_gridsearch_plot(model_gs, "max_depth", "max depth of the tree", "Decision Tree")
#
#
# # finding the maximum to test it
# etmodel = model_gs.best_estimator_
# etmodel.fit(X_train, Y_train)
# Y_train_pred = etmodel.predict(X_train)
# Y_test_pred = etmodel.predict(X_test)
#
#
# # calculate cmtr, acctr, cmte, accte
# metrics_dict = calculate_metrics(Y_train, Y_test, Y_train_pred, Y_test_pred)
#
#
# # Visualize Confusion Matrix
# confusion_matrix_plotter(etmodel, X_test, Y_test)
#
#
# # plot classification report
# print('Classification Report \n', classification_report(Y_test, Y_test_pred))
#
#
# # calculate precision & recall metrics
# metrics_dict.update(calculate_precision_recall(Y_test, Y_test_pred))
#
#
# # calculate f1 score
# metrics_dict["f1te"] = calculate_f1_score(Y_test, Y_test_pred)
#
#
# # calculate ROC and AUC and plot the curve
# roc_auc_results = calculate_roc_auc(etmodel, X_test, Y_test)
# metrics_dict["roc_auc"] = roc_auc_results["roc_auc"]
# metrics_dict["fpr"] = roc_auc_results["fpr"]
# metrics_dict["tpr"] = roc_auc_results["tpr"]
# plot_model_metrics(metrics_dict, 'ROC Curve of Decision Tree')
#
#
# # plot feature importance
# plot_feature_importance(etmodel, 'Decision Tree - Feature Importance')
#
# # add metrics to the report
# report.loc[len(report)] = ['Decision Tree', model_gs.best_params_, metrics_dict["acctr"], metrics_dict["accte"], metrics_dict["f1te"], metrics_dict["roc_auc"], metrics_dict["precision"], metrics_dict["recall"]]
#
#
# # testing
# print(report)
#
# results = pd.DataFrame(model_gs.cv_results_)
# results.to_csv('Testing/'+source+'_decision_tree_report.csv', index = False)
#
#
#
# # #=============================================================================
# # #show tree using graphviz
# # import graphviz
# # dot_data = sk.tree.export_graphviz(etmodel, out_file=None,
# #                           feature_names=list(X),
# #                           filled=True, rounded=True,
# #                           special_characters=True)
# # graph = graphviz.Source(dot_data)
# # graph.format = 'png'
# # graph.render("Churn_entropy")
# # #=============================================================================
#
# #################
# #     Gini      #
# #################
#
#
# # Build Gini model where the max_depth is tested
# # new GridSearchCV implementation
# # attention: the parameter 'n_jobs=-1' means all CPU cores will be used. This may result in an overflow of the RAM
# # do not forget to change parameter 'cv'
# model_gs_param_grid = {"max_depth": [2,3,4,5,6,7]}
# model_gs = sk.model_selection.GridSearchCV(estimator=DecisionTreeClassifier(random_state=0, class_weight='balanced'), param_grid=model_gs_param_grid,
#                                            scoring='accuracy', cv=10, n_jobs=-1)
# model_gs.fit(X_train, Y_train)
# print(model_gs.cv_results_)
#
# results = pd.DataFrame(model_gs.cv_results_)
# results.to_csv('Testing/'+source+'_gini_report.csv', index = False)
#
# # creating a Plot
# create_gridsearch_plot(model_gs, "max_depth", "max depth of the tree", "Gini Decision Tree")
#
#
# # finding the maximum to test it
# gtmodel = model_gs.best_estimator_
# gtmodel.fit(X_train, Y_train)
# Y_train_pred = gtmodel.predict(X_train)
# Y_test_pred = gtmodel.predict(X_test)
#
# # calculate cmtr, acctr, cmte, accte
# metrics_dict = calculate_metrics(Y_train, Y_test, Y_train_pred, Y_test_pred)
#
#
# # Visualize Confusion Matrix
# confusion_matrix_plotter(gtmodel, X_test, Y_test)
#
#
# # print classification report
# print('Classification Report \n', classification_report(Y_test, Y_test_pred))
#
#
# # calculate precision & recall metrics
# metrics_dict.update(calculate_precision_recall(Y_test, Y_test_pred))
#
#
# # calculate f1 score
# metrics_dict["f1te"] = calculate_f1_score(Y_test, Y_test_pred)
#
#
# # calculate ROC and AUC and plot the curve
# roc_auc_results = calculate_roc_auc(gtmodel, X_test, Y_test)
# metrics_dict["roc_auc"] = roc_auc_results["roc_auc"]
# metrics_dict["fpr"] = roc_auc_results["fpr"]
# metrics_dict["tpr"] = roc_auc_results["tpr"]
# plot_model_metrics(metrics_dict, 'ROC Curve of Gini Decision Tree')
#
#
# # plot feature importance
# plot_feature_importance(gtmodel, 'Gini Decision Tree - Feature Importance')
#
# # add metrics to the report
# report.loc[len(report)] = ['Gini', model_gs.best_params_, metrics_dict["acctr"], metrics_dict["accte"], metrics_dict["f1te"], metrics_dict["roc_auc"], metrics_dict["precision"], metrics_dict["recall"]]
#
#
# # testing
# print(report)
#
#
# results = pd.DataFrame(model_gs.cv_results_)
# results.to_csv('Testing/'+source+'_gini_report.csv', index = False)
#

#################
# Random Forest #
#################

from sklearn.ensemble import RandomForestClassifier


# Build RandomForest model where the max_depth and the n_estimators is tested
# new GridSearchCV implementation
# attention: the parameter 'n_jobs=-1' means all CPU cores will be used. This may result in an overflow of the RAM
# do not forget to change parameter 'cv'
#model_gs_param_grid = {'max_depth': [12, 13, 14, 15, 16, 17, 18], 'n_estimators': [200, 400, 500, 600]}
model_gs_param_grid = {'max_depth': [6, 7, 8, 9, 10, 11, 12, 13]}
model_gs = sk.model_selection.GridSearchCV(estimator=RandomForestClassifier(random_state=0, class_weight='balanced', n_estimators=200), param_grid=model_gs_param_grid,
                                           scoring='accuracy', cv=10, n_jobs=-1)
model_gs.fit(X_train, Y_train)
print('Random-Forest-Results-Training: /n:', model_gs.cv_results_)


# # print GridSearchCV results
# create_tabulate_plot(model_gs, "max_depth", "n_estimators", "max depth", "number of trees", "Random Forest")

# creating a Plot
create_gridsearch_plot(model_gs, "max_depth", "max depth of the tree", "Random Forest")


# finding the maximum to test it
rfmodel = model_gs.best_estimator_
rfmodel.fit(X_train, Y_train)
Y_train_pred = rfmodel.predict(X_train)
Y_test_pred = rfmodel.predict(X_test)

# calculate cmtr, acctr, cmte, accte
metrics_dict = calculate_metrics(Y_train, Y_test, Y_train_pred, Y_test_pred)


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



#################################
# NEEDS TO BE THE LAST MODEL!!  #
#################################
# LightGBM  #
#############

# USES THE ORIGINAL DATASET WITHOUT DURATION
source = "BankMarketing"

data = pd.read_csv(source+'.csv')

data = data.drop(columns='duration')
data[data.select_dtypes('object').columns.tolist()] = data[data.select_dtypes('object').columns.tolist()].astype('category')


# JUST FOR TESTING
#data = data.copy().iloc[:1000,:]

# creating X and Y categories
X = data.copy().drop(columns='y')
Y = data['y']


from sklearn.model_selection import train_test_split

# raised to 0.8
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


import lightgbm as lgb

#print("object columns: \n", X_train.select_dtypes('object').columns.tolist())
#print("category columns:\n", X_train.select_dtypes('category').columns.tolist())
#print("columns:\n", X_train.columns)
#not used yet
cat_col = X_train.select_dtypes('object').columns.tolist() + X_train.select_dtypes('category').columns.tolist()

#X_train[cat_col] = X_train[cat_col].astype('category')
#X_train[cat_col] = X_train[cat_col].astype('category')

d_train = lgb.Dataset(X_train, label=Y_train)
d_valid = lgb.Dataset(X_test, label=Y_test)
#d_train = lgb.Dataset(X_train, label=Y_train, feature_name=X_train.columns.tolist(), categorical_feature=cat_col)

print(X_train.columns.tolist())

model_gs_param_grid = {"max_depth": [3,4,5,6,7,8,10,12,15,17,20], "learning_rate": [0.005, 0.01, 0.05, 0.1, 0.2], "num_leaves": [7, 15, 31, 63, 127, 255], "num_iteration": [300, 500, 700]}
# parameter for lgb.LGBMClassifier(feature_name=X_train.columns.tolist())
model_gs = sk.model_selection.GridSearchCV(estimator=lgb.LGBMClassifier(random_state=0, objective='binary', class_weight='balanced'), param_grid=model_gs_param_grid,
                                           scoring='accuracy', cv=10, n_jobs=-1)
model_gs.fit(X_train, Y_train)


# finding the maximum to test it
lgbmodel = model_gs.best_estimator_
lgbmodel.fit(X_train, Y_train)
Y_train_pred = lgbmodel.predict(X_train)
Y_test_pred = lgbmodel.predict(X_test)


# calculate cmtr, acctr, cmte, accte
metrics_dict = calculate_metrics(Y_train, Y_test, Y_train_pred, Y_test_pred)


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



#############################
# PRINT THE FINAL REPORT    #
#############################

report.to_csv('Testing/report_all'+source+'.csv', index=False)

print("stop")