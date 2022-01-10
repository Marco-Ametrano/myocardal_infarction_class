#RANDOM FOREST WITH Y_UNICA
model= RandomForestClassifier(n_estimators=100, random_state=22)
model.fit(X_compl_train, Y_unica_train.values.ravel())
importances=model.feature_importances_
df_imp=pd.DataFrame({'Features':X_compl_train.columns, 'importances':importances})
df_imp.plot(kind='bar')

np.where(importances > 0.01)
X_train_rid = X_compl_train.iloc[:, [0,   1,   2,   3,   4,   5,   6,   8,   9,  24,  32,  33,  40,
         41,  42,  43,  45,  78,  79,  81,  82,  83,  84,  85,  86,  87,
         90,  91,  94,  97,  99, 100, 101]] 
X_test_rid = X_compl_test.iloc[:, [ 0,   1,   2,   3,   4,   5,   6,   8,   9,  24,  32,  33,  40,
         41,  42,  43,  45,  78,  79,  81,  82,  83,  84,  85,  86,  87,
         90,  91,  94,  97,  99, 100, 101]]

param = dict(n_estimators = [100], max_depth =range(5,10),  
              min_samples_split = range(2,20), 
             min_samples_leaf = range(2,15))
grid_search_cv = GridSearchCV(RandomForestClassifier(random_state=22), param, verbose=1, cv=3, n_jobs=-1)
grid_search_cv.fit(X_train_rid, Y_unica_train)
print(grid_search_cv.best_params_)
rf=RandomForestClassifier(max_depth= 6, min_samples_leaf= 3, min_samples_split= 7, n_estimators= 100, random_state=22)
rf.fit(X_train_rid,Y_unica_train.values.ravel())

y_train_pred = rf.predict(X_train_rid)
print(classification_report(Y_unica_train, y_train_pred))
y_test_pred = rf.predict(X_test_rid) 
print(classification_report(Y_unica_test, y_test_pred))

#RANDOM FOREST WITH Y_UNICA-WEIGHTS=BALANCED
model= RandomForestClassifier(n_estimators=100, random_state=22, class_weight="balanced")
model.fit(X_compl_train, Y_unica_train.values.ravel())
importances=model.feature_importances_
df_imp=pd.DataFrame({'Features':X_compl_train.columns, 'importances':importances})
df_imp.plot(kind='bar')

np.where(importances > 0.01)
X_train_rid = X_compl_train.iloc[:, [ 0,   1,   2,   3,   4,   5,   6,   8,  32,  33,  40,  41,  42,
         43,  45,  53,  78,  79,  81,  82,  83,  84,  85,  86,  87,  90,
         91,  94,  97,  99, 100]]
X_test_rid = X_compl_test.iloc[:, [ 0,   1,   2,   3,   4,   5,   6,   8,  32,  33,  40,  41,  42,
         43,  45,  53,  78,  79,  81,  82,  83,  84,  85,  86,  87,  90,
         91,  94,  97,  99, 100]]

param = dict(n_estimators = [100], max_depth =range(5,10),  
             min_samples_split = range(2,20), 
            min_samples_leaf = range(2,15))
grid_search_cv = GridSearchCV(RandomForestClassifier(random_state=22, class_weight="balanced"), param, verbose=1, cv=3, n_jobs=-1)
grid_search_cv.fit(X_train_rid, Y_unica_train.values.ravel())
print(grid_search_cv.best_params_)

rf=RandomForestClassifier(max_depth= 9, min_samples_leaf= 2, min_samples_split= 2, n_estimators= 100, random_state=22, 
                          class_weight="balanced")
rf.fit(X_train_rid,Y_unica_train.values.ravel())
y_train_pred = rf.predict(X_train_rid) 
print(classification_report(Y_unica_train, y_train_pred))

y_test_pred = rf.predict(X_test_rid) 
print(classification_report(Y_unica_test, y_test_pred))

#XGBOOST WITH Y_UNICA
XGBclassifier= xgboost.XGBClassifier(random_state=22)
XGBclassifier.fit(X_compl_train, Y_unica_train.values.ravel())
importances=XGBclassifier.feature_importances_; importances

np.where(importances>0.012)
X_train_rid = X_compl_train.iloc[:, [0,  3,  5,  6,  8,  9, 27, 28, 29, 32, 34, 40, 41, 42, 44, 46, 47,
        54, 55, 64, 65, 69, 70, 72, 73, 78, 81, 82, 83, 86, 89, 90, 95, 96,
        97, 98]]
X_test_rid = X_compl_test.iloc[:, [0,  3,  5,  6,  8,  9, 27, 28, 29, 32, 34, 40, 41, 42, 44, 46, 47,
        54, 55, 64, 65, 69, 70, 72, 73, 78, 81, 82, 83, 86, 89, 90, 95, 96,
        97, 98]]

XGBparams = dict(eta=[0.05, 0.10, 0.15, 0.20], gamma=[ 0.0, 0.1, 0.2 , 0.3],
                 max_depth=[ 3, 4, 5, 6, 8, 10], min_child_weight=[ 1, 3, 5],
                 colsample_bytree=[ 0.3, 0.4, 0.5])
grid_search_XGB = GridSearchCV(XGBclassifier, XGBparams,verbose=1, cv=3, n_jobs=-1)
grid_search_XGB.fit(X_train_rid,Y_unica_train.values.ravel())
print(grid_search_XGB.best_params_)
XGBclassifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.5, gamma=0.3, eta=0.05,
       max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='multi:softmax',num_class=12, random_state=22, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)
XGBclassifier.fit(X_train_rid,Y_unica_train.values.ravel())

y_train_pred=XGBclassifier.predict(X_train_rid)
print(classification_report(Y_unica_train, y_train_pred))
y_test_pred = XGBclassifier.predict(X_test_rid)
print(classification_report(Y_unica_test, y_test_pred))
