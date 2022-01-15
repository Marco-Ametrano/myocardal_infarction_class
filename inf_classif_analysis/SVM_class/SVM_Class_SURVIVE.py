#SVM WITH ALL THE VARIABLES
parameters = {'kernel':('linear', 'rbf',"poly","sigmoid"), 'C':range(1, 10)}
svc = svm.SVC(random_state=22, class_weight="balanced")
grid_svm= GridSearchCV(svc, parameters,cv=5)
grid_svm.fit(X_train, Y_survive_train.values.ravel())
print(grid_svm.best_params_)

Support_class= svm.SVC(C=2,kernel="linear",random_state=22, class_weight="balanced")
Support_class.fit(X_train, Y_survive_train.values.ravel())
y_pred1= Support_class.predict(X_train)
print(classification_report(Y_survive_train,y_pred1))
y_pred2= Support_class.predict(X_test)
print(classification_report(Y_survive_test, y_pred2))

#SVM WITH FEATURE SELECTION
fs = SelectKBest(score_func = mutual_info_classif)
fs.fit(X_train, Y_survive_train)
for i in range(len(fs.scores_)):
  print("Feature %d: %f" % (i, fs.scores_[i]))
 
np.where(fs.scores_>0.010)
X_train_rid = X_train.iloc[:, [ 0,   2,   3,  23,  32,  33,  44,  45,  56,  72,  81,  84,  89,
         90, 100, 101, 102]]
X_test_rid = X_test.iloc[:, [ 0,   2,   3,  23,  32,  33,  44,  45,  56,  72,  81,  84,  89,
         90, 100, 101, 102]]
parameters = {'kernel':('linear', 'rbf',"poly","sigmoid"), 'C':range(1, 10)}
svc = svm.SVC(random_state=22, class_weight="balanced")
grid_svm= GridSearchCV(svc, parameters,cv=5)
grid_svm.fit(X_train_rid, Y_survive_train.values.ravel())
print(grid_svm.best_params_)

Support_class= svm.SVC(C=1,kernel="sigmoid",random_state=22, class_weight="balanced")
Support_class.fit(X_train_rid, Y_survive_train.values.ravel())

y_pred1= Support_class.predict(X_train_rid)
print(classification_report(Y_survive_train,y_pred1))
y_pred2= Support_class.predict(X_test_rid)
print(classification_report(Y_survive_test, y_pred2))
