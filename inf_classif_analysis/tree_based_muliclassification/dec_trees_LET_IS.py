#DECISION TREES WITH LET_IS
X_train, X_test, Y_LET_train, Y_LET_test = train_test_split(X, LET_IS, test_size=0.3, random_state= 22)
model_imp = DecisionTreeClassifier(random_state=22)
model_imp.fit(X_train, Y_LET_train)
importance = model_imp.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

np.where(importance>0.015)
X_train_rid = X_train.iloc[:, [0,  1,  3,  5,  6,  8,  9, 32, 40, 41, 43, 44, 53, 54, 70, 79, 81,
        83, 85, 89, 90, 93, 94, 95, 96]]
X_test_rid = X_test.iloc[:, [0,  1,  3,  5,  6,  8,  9, 32, 40, 41, 43, 44, 53, 54, 70, 79, 81,
        83, 85, 89, 90, 93, 94, 95, 96]]

class_tree=DecisionTreeClassifier(random_state=22)
param = {'max_depth':range(1, 6), 'min_samples_split':range(3,20), 'min_samples_leaf':range(2,15)} 
grid = GridSearchCV(class_tree, param, cv=3)
grid.fit(X_train_rid, Y_LET_train)
print(grid.best_params_)
mod_tree=DecisionTreeClassifier(max_depth=3,min_samples_split=14,min_samples_leaf=3,random_state=22)
mod_tree.fit(X_train_rid, Y_LET_train)
text_representation = tree.export_text(mod_tree)
print(text_representation)
fig1 = plt.figure(figsize=(25,20))
_ = tree.plot_tree(mod_tree,feature_names=list(X_train_rid.columns), class_names=['0', '1', '2', '3', '4', '5', '6', '7'],filled=True)

y_train_pred = mod_tree.predict(X_train_rid)
print(classification_report(Y_LET_train, y_train_pred))
y_test_pred = mod_tree.predict(X_test_rid)
print(classification_report(Y_LET_test, y_test_pred))
