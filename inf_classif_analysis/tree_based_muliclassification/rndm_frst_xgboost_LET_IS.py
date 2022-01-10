#RANDOM FOREST WITH LET_IS
model= RandomForestClassifier(n_estimators=100, random_state=22)
model.fit(X_train, Y_LET_train.values.ravel())
importances=model.feature_importances_
df_imp=pd.DataFrame({'Features':X_train.columns, 'importances':importances})
df_imp.plot(kind='bar')
np.where(importances > 0.015)
X_train_rid = X_train.iloc[:, [0,  2,  3,  4,  5,  8,  9, 32, 33, 40, 41, 42, 79, 81, 82, 83, 84,
        85, 86, 87, 90, 91]]
X_test_rid = X_test.iloc[:, [0,  2,  3,  4,  5,  8,  9, 32, 33, 40, 41, 42, 79, 81, 82, 83, 84,
        85, 86, 87, 90, 91]]

param = dict(n_estimators = [100], max_depth =range(5,10),  
              min_samples_split = range(2,20), 
             min_samples_leaf = range(2,15))
grid_search_cv = GridSearchCV(RandomForestClassifier(random_state=22), param, verbose=1, cv=3, n_jobs=-1)
grid_search_cv.fit(X_train_rid, Y_LET_train)
print(grid_search_cv.best_params_)
rf=RandomForestClassifier(max_depth= 5, min_samples_leaf= 2, min_samples_split= 2, n_estimators= 100, random_state=22)
rf.fit(X_train_rid,Y_LET_train.values.ravel())

y_train_pred = rf.predict(X_train_rid)
print(classification_report(Y_LET_train, y_train_pred))
y_test_pred = rf.predict(X_test_rid) 
print(classification_report(Y_LET_test, y_test_pred))

#XGBOOST WITH LET_IS
XGBclassifier= xgboost.XGBClassifier(random_state=22)
XGBclassifier.fit(X_train, Y_LET_train.values.ravel())
importances=XGBclassifier.feature_importances_; importances

np.where(importances>0.015)
X_train_rid = X_train.iloc[:, [0,  3,  4,  5,  8,  9, 13, 24, 29, 32, 36, 40, 41, 44, 46, 49, 53,
        54, 55, 61, 65, 70, 85, 86, 87, 89, 90, 93, 94 ]]
X_test_rid = X_test.iloc[:, [0,  3,  4,  5,  8,  9, 13, 24, 29, 32, 36, 40, 41, 44, 46, 49, 53,
        54, 55, 61, 65, 70, 85, 86, 87, 89, 90, 93, 94]]

XGBparams = dict(eta=[0.05, 0.10, 0.15, 0.20], gamma=[ 0.0, 0.1, 0.2 , 0.3],
                 max_depth=[ 3, 4, 5, 6, 8, 10], min_child_weight=[ 1, 3, 5],
                 colsample_bytree=[ 0.3, 0.4, 0.5])
grid_search_XGB = GridSearchCV(XGBclassifier, XGBparams,verbose=1, cv=3, n_jobs=-1)
grid_search_XGB.fit(X_train_rid,Y_LET_train.values.ravel())
print(grid_search_XGB.best_params_)

XGBclassifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.5, gamma=0.3, eta=0.05,
       max_delta_step=0, max_depth=8, min_child_weight=1, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='multi:softmax',num_class=12, random_state=22, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)

XGBclassifier.fit(X_train_rid,Y_LET_train.values.ravel())
y_train_pred=XGBclassifier.predict(X_train_rid)
print(classification_report(Y_LET_train, y_train_pred))
y_test_pred = XGBclassifier.predict(X_test_rid)
print(classification_report(Y_LET_test, y_test_pred))
