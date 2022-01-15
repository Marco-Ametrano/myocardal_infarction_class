#XGBoost ZSN no scale_pos_weight
XGBclassifier= xgboost.XGBClassifier(random_state=22)
XGBclassifier.fit(X_train, Y_ZSN_train.values.ravel())
importances=XGBclassifier.feature_importances_; importances
np.where(importances>0.02)
X_train_rid = X_train.iloc[:, [  0,   9,  24,  27,  33,  41,  43,  44,  46,  53,  78,  85,  87,
         91,  92,  93,  97,  98, 101]]
X_test_rid = X_test.iloc[:, [  0,   9,  24,  27,  33,  41,  43,  44,  46,  53,  78,  85,  87,
         91,  92,  93,  97,  98, 101]]

XGBparams = dict(eta=[0.05, 0.10, 0.15, 0.20], gamma=[ 0.0, 0.1, 0.2 , 0.3],
                 max_depth=[ 3, 4, 5, 6, 8, 10], min_child_weight=[ 1, 3, 5],
                 colsample_bytree=[ 0.3, 0.4, 0.5])
grid_search_XGB = GridSearchCV(XGBclassifier, XGBparams,verbose=1, cv=3, n_jobs=-1)
grid_search_XGB.fit(X_train_rid,Y_ZSN_train.values.ravel())
print(grid_search_XGB.best_params_)
XGBclassifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.4, gamma=0.2, eta=0.05,
       max_delta_step=0, max_depth=5, min_child_weight=1, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='multi:softmax',num_class=12, random_state=22, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)

XGBclassifier.fit(X_train_rid,Y_ZSN_train.values.ravel())
y_train_pred=XGBclassifier.predict(X_train_rid)
print(classification_report(Y_ZSN_train, y_train_pred))

y_test_pred = XGBclassifier.predict(X_test_rid)
print(classification_report(Y_ZSN_test, y_test_pred))

#XGBoost ZSN scale_pos_weight

ZSN.value_counts()
XGBclassifier= xgboost.XGBClassifier(random_state=22, scale_pos_weight=2.99)
XGBclassifier.fit(X_train, Y_ZSN_train.values.ravel())
importances=XGBclassifier.feature_importances_; importances
np.where(importances>0.02)

X_train_rid = X_train.iloc[:, [  0,  1,  9, 24, 27, 40, 41, 44, 46, 49, 53, 85, 86, 89, 91, 92, 93,
        98, 99]]
X_test_rid = X_test.iloc[:, [  0,  1,  9, 24, 27, 40, 41, 44, 46, 49, 53, 85, 86, 89, 91, 92, 93,
        98, 99]]

XGBparams = dict(eta=[0.05, 0.10, 0.15, 0.20], gamma=[ 0.0, 0.1, 0.2 , 0.3],
                 max_depth=[ 3, 4, 5, 6, 8, 10], min_child_weight=[ 1, 3, 5],
                 colsample_bytree=[ 0.3, 0.4, 0.5])
grid_search_XGB = GridSearchCV(XGBclassifier, XGBparams,verbose=1, cv=3, n_jobs=-1)
grid_search_XGB.fit(X_train_rid,Y_ZSN_train.values.ravel())
print(grid_search_XGB.best_params_)

XGBclassifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.4, gamma=0.0, eta=0.05,
       max_delta_step=0, max_depth=8, min_child_weight=1, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='multi:softmax',num_class=12, random_state=22, reg_alpha=0,
       reg_lambda=1, scale_pos_weight= 2.99, seed=None, silent=True,
       subsample=1)
XGBclassifier.fit(X_train_rid,Y_ZSN_train.values.ravel())
y_train_pred=XGBclassifier.predict(X_train_rid)
print(classification_report(Y_ZSN_train, y_train_pred))

y_test_pred = XGBclassifier.predict(X_test_rid)
print(classification_report(Y_ZSN_test, y_test_pred))
