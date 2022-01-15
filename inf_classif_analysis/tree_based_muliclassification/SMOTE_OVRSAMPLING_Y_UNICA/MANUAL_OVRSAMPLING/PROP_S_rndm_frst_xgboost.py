#Random Forest SMOTE PROP Y_UNICA
model= RandomForestClassifier(n_estimators=100, random_state=22)
model.fit(X_Smote_train,Y_Smote_train.values.ravel())
importances=model.feature_importances_
feat_importances = pd.Series(model.feature_importances_, index=X_Smote_train.columns)
df_imp=pd.DataFrame({'Features':X_Smote_train.columns, 'importances':importances})
feat_importances.nlargest(40).plot(kind='barh')
plt.show()
df_imp.sort_values('importances',ascending = False)

np.where(importances>0.015)
X_Smote_train_rid = X_Smote_train.iloc[:, [0,  1,  2,  3,  4,  5,  6,  8,  9, 32, 33, 40, 41, 42, 78, 79, 81,
        84, 85, 86, 90, 91, 97]]
X_test_rid = X_compl_test.iloc[:, [0,  1,  2,  3,  4,  5,  6,  8,  9, 32, 33, 40, 41, 42, 78, 79, 81,
        84, 85, 86, 90, 91, 97]]

param = dict(n_estimators = [100], max_depth =range(5,10),  
             min_samples_split = range(2,30), 
            min_samples_leaf = range(2,20))
grid_search_cv = GridSearchCV(RandomForestClassifier(random_state=22), param, verbose=1, cv=5, n_jobs=-1)
grid_search_cv.fit(X_Smote_train_rid, Y_Smote_train)
print(grid_search_cv.best_params_)
rf=RandomForestClassifier(max_depth= 9, min_samples_leaf= 2, min_samples_split= 2, n_estimators= 100, random_state=22)
rf.fit(X_Smote_train_rid,Y_Smote_train.values.ravel())
y_train_pred = rf.predict(X_Smote_train_rid)
print(classification_report(Y_Smote_train, y_train_pred))
y_test_pred = rf.predict(X_test_rid)
print(classification_report(Y_unica_test, y_test_pred))

#XGBoost SMOTE PROP Y_UNICA
XGBclassifier= xgboost.XGBClassifier(random_state=22)

XGBclassifier.fit(X_Smote_train, Y_Smote_train.values.ravel())
importances=XGBclassifier.feature_importances_; importances
np.where(importances>0.012)

X_Smote_train_rid = X_Smote_train.iloc[:, [0,  4,  6,  9, 13, 14, 21, 24, 28, 32, 33, 34, 36, 40, 41, 42, 44,
        47, 51, 53, 55, 56, 57, 68, 69, 70, 72, 73, 78, 79, 86, 89, 90, 91,
        97]] 
X_test_rid = X_compl_test.iloc[:, [0,  4,  6,  9, 13, 14, 21, 24, 28, 32, 33, 34, 36, 40, 41, 42, 44,
        47, 51, 53, 55, 56, 57, 68, 69, 70, 72, 73, 78, 79, 86, 89, 90, 91,
        97]]

XGBparams = dict(eta=[0.05, 0.10, 0.15, 0.20], gamma=[ 0.0, 0.1, 0.2 , 0.3],
                 max_depth=[ 3, 4, 5, 6, 8, 10], min_child_weight=[ 1, 3, 5],
                 colsample_bytree=[ 0.3, 0.4, 0.5])

grid_search_XGB = GridSearchCV(XGBclassifier, XGBparams,verbose=1, cv=3, n_jobs=-1)
grid_search_XGB.fit(X_Smote_train_rid,Y_Smote_train.values.ravel())
print(grid_search_XGB.best_params_)

XGBclassifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.4, gamma=0.1, eta=0.05,
       max_delta_step=0, max_depth=10, min_child_weight=1, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='multi:softmax',num_class=12, random_state=22, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)

XGBclassifier.fit(X_Smote_train_rid,Y_Smote_train.values.ravel())
y_train_pred=XGBclassifier.predict(X_Smote_train_rid)
print(classification_report(Y_Smote_train, y_train_pred))
y_test_pred = XGBclassifier.predict(X_test_rid)
print(classification_report(Y_unica_test, y_test_pred))
