model= RandomForestClassifier(n_estimators=100, random_state=22, class_weight="balanced")
model.fit(X_train, Y_ZSN_train.values.ravel())
importances=model.feature_importances_
df_imp=pd.DataFrame({'Features':X_train.columns, 'importances':importances})
df_imp.plot(kind='bar')

np.where(importances > 0.015)
X_train_rid = X_train.iloc[:, [0,  2,  3,  5,  6,  8,  9, 32, 33, 40, 41, 42, 79, 81, 82, 83, 84,
        85, 86, 91]]
X_test_rid = X_test.iloc[:, [ 0,  2,  3,  5,  6,  8,  9, 32, 33, 40, 41, 42, 79, 81, 82, 83, 84,
        85, 86, 91]]
param = dict(n_estimators = [100], max_depth =range(5,10),  
             min_samples_split = range(2,20), 
            min_samples_leaf = range(2,15))
grid_search_cv = GridSearchCV(RandomForestClassifier(random_state=22, class_weight= "balanced"), param, verbose=1, cv=5, n_jobs=-1)
grid_search_cv.fit(X_train_rid, Y_ZSN_train)
print(grid_search_cv.best_params_)

rf=RandomForestClassifier(max_depth= 9, min_samples_leaf= 2, min_samples_split= 2, n_estimators= 100, random_state=22, class_weight="balanced")
rf.fit(X_train_rid,Y_ZSN_train.values.ravel())
y_train_pred = rf.predict(X_train_rid)
print(classification_report(Y_ZSN_train, y_train_pred))

y_test_pred = rf.predict(X_test_rid) 
print(classification_report(Y_ZSN_test, y_test_pred))
