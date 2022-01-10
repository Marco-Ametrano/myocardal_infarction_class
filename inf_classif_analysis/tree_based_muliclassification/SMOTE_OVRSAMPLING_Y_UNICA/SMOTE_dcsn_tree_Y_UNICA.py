model_imp = DecisionTreeClassifier(random_state=22)
model_imp.fit(X_Smote_train,Y_Smote_train)
importance = model_imp.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
plt.bar([x for x in range(len(importance))], importance)
plt.show()

np.where(importance>0.015)
X_Smote_train_rid = X_Smote_train.iloc[:, [0,   2,   3,   4,   5,   6,   8,   9,  27,  28,  32,  33,  41,
         42,  44,  45,  53,  64,  78,  79,  81,  84,  85,  86,  87,  90,
         91,  95,  97, 100]]
X_test_rid = X_compl_test.iloc[:, [0,   2,   3,   4,   5,   6,   8,   9,  27,  28,  32,  33,  41,
         42,  44,  45,  53,  64,  78,  79,  81,  84,  85,  86,  87,  90,
         91,  95,  97, 100]]

class_tree=DecisionTreeClassifier(random_state=22)
param = {'max_depth':range(1, 6), 'min_samples_split':range(2,30), 'min_samples_leaf':range(2,25)} 
grid = GridSearchCV(class_tree, param, cv=5)
grid.fit(X_Smote_train_rid, Y_Smote_train)
print(grid.best_params_)

mod_tree=DecisionTreeClassifier(max_depth=5,min_samples_split=8,min_samples_leaf=2,random_state=22)
mod_tree.fit(X_Smote_train_rid, Y_Smote_train)
text_representation = tree.export_text(mod_tree)
print(text_representation)
fig1 = plt.figure(figsize=(25,20))
_ = tree.plot_tree(mod_tree,feature_names=list(X_Smote_train_rid.columns),filled=True)

y_train_pred = mod_tree.predict(X_Smote_train_rid)
print(classification_report(Y_Smote_train, y_train_pred))
y_test_pred = mod_tree.predict(X_test_rid)
print(classification_report(Y_unica_test, y_test_pred))
