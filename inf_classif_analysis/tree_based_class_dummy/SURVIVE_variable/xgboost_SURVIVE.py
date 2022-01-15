#XGBoost Survive no scale_pos_weight
XGBclassifier= xgboost.XGBClassifier(random_state=22)
XGBclassifier.fit(X_train, Y_survive_train.values.ravel())
importances=XGBclassifier.feature_importances_; importances

np.where(importances>0.015)
X_train_rid = X_train.iloc[:, [ 0,   3,   4,   5,   6,   8,   9,  24,  28,  32,  33,  40,  41,
         42,  44,  45,  53,  54,  55,  70,  79,  81,  83,  84,  85,  86,
         90,  91,  93,  96,  98, 103]]
X_test_rid = X_test.iloc[:, [ 0,   3,   4,   5,   6,   8,   9,  24,  28,  32,  33,  40,  41,
         42,  44,  45,  53,  54,  55,  70,  79,  81,  83,  84,  85,  86,
         90,  91,  93,  96,  98, 103]]
XGBparams = dict(eta=[0.05, 0.10, 0.15, 0.20], gamma=[ 0.0, 0.1, 0.2 , 0.3],
                 max_depth=[ 3, 4, 5, 6, 8, 10], min_child_weight=[ 1, 3, 5],
                 colsample_bytree=[ 0.3, 0.4, 0.5])

grid_search_XGB = GridSearchCV(XGBclassifier, XGBparams,verbose=1, cv=5, n_jobs=-1)
grid_search_XGB.fit(X_train_rid,Y_survive_train.values.ravel())
print(grid_search_XGB.best_params_)

XGBclassifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.4, gamma=0.1, eta=0.05,
       max_delta_step=0, max_depth=6, min_child_weight=5, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='multi:softmax',num_class=12, random_state=22, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)
XGBclassifier.fit(X_train_rid,Y_survive_train.values.ravel())
y_train_pred=XGBclassifier.predict(X_train_rid)
print(classification_report(Y_survive_train, y_train_pred))

y_test_pred = XGBclassifier.predict(X_test_rid)
print(classification_report(Y_survive_test, y_test_pred))

#XGBoost Survive scale_pos_weight

survive.value_counts()
XGBclassifier= xgboost.XGBClassifier(random_state=22, scale_pos_weight=0.11)
XGBclassifier.fit(X_train, Y_survive_train.values.ravel())
importances=XGBclassifier.feature_importances_; importances

np.where(importances>0.015)
X_train_rid = X_train.iloc[:, [ 0,  1,  3,  4,  5,  6,  8,  9, 24, 32, 40, 41, 42, 45, 49, 70, 79,
        81, 83, 84, 85, 86, 90, 91, 99]]
X_test_rid = X_test.iloc[:, [0,  1,  3,  4,  5,  6,  8,  9, 24, 32, 40, 41, 42, 45, 49, 70, 79,
        81, 83, 84, 85, 86, 90, 91, 99]]

XGBparams = dict(eta=[0.05, 0.10, 0.15, 0.20], gamma=[ 0.0, 0.1, 0.2 , 0.3],
                 max_depth=[ 3, 4, 5, 6, 8, 10], min_child_weight=[ 1, 3, 5],
                 colsample_bytree=[ 0.3, 0.4, 0.5])
grid_search_XGB = GridSearchCV(XGBclassifier, XGBparams,verbose=1, cv=5, n_jobs=-1)
grid_search_XGB.fit(X_train_rid,Y_survive_train.values.ravel())
print(grid_search_XGB.best_params_)

XGBclassifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.3, gamma=0.1, eta=0.05,
       max_delta_step=0, max_depth=6, min_child_weight=1, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='multi:softmax',num_class=12, random_state=22, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=0.11, seed=None, silent=True,
       subsample=1)
XGBclassifier.fit(X_train_rid,Y_survive_train.values.ravel())

y_train_pred=XGBclassifier.predict(X_train_rid)
print(classification_report(Y_survive_train, y_train_pred))
y_test_pred = XGBclassifier.predict(X_test_rid)
print(classification_report(Y_survive_test, y_test_pred))


