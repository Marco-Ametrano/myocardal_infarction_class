X_train, X_test, Y_ZSN_train, Y_ZSN_test = train_test_split(X, ZSN, test_size=0.3, random_state= 22)
model_imp = DecisionTreeClassifier(random_state=22, class_weight="balanced")
model_imp.fit(X_train, Y_ZSN_train)
importance = model_imp.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
plt.bar([x for x in range(len(importance))], importance)
plt.show()

np.where(importance>0.015)
X_train_rid = X_train.iloc[:, [ 0,  2,  8,  9, 27, 28, 32, 33, 40, 41, 49, 79, 81, 82, 83, 84, 85,
        86, 94]]
X_test_rid = X_test.iloc[:, [ 0,  2,  8,  9, 27, 28, 32, 33, 40, 41, 49, 79, 81, 82, 83, 84, 85,
        86, 94]]

class_tree=DecisionTreeClassifier(random_state=22, class_weight="balanced")
param = {'max_depth':range(1, 6), 'min_samples_split':range(2,20), 'min_samples_leaf':range(2,15)} 
grid = GridSearchCV(class_tree, param, cv=5)
grid.fit(X_train_rid, Y_ZSN_train)
print(grid.best_params_)

mod_tree=DecisionTreeClassifier(max_depth=1,min_samples_split=2,min_samples_leaf=2,random_state=22)
mod_tree.fit(X_train_rid, Y_ZSN_train)
text_representation = tree.export_text(mod_tree)
print(text_representation)

fig1 = plt.figure(figsize=(5,5))
_ = tree.plot_tree(mod_tree,feature_names=list(X_train_rid.columns),filled=True)

y_train_pred = mod_tree.predict(X_train_rid)
print(classification_report(Y_ZSN_train, y_train_pred))

y_test_pred = mod_tree.predict(X_test_rid)
print(classification_report(Y_ZSN_test, y_test_pred))
