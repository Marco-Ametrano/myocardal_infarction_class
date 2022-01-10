#DECISION TREE Y_UNICA-NO BALANCED
X_compl_train, X_compl_test, Y_unica_train, Y_unica_test = train_test_split(X_compl, Y_unica, test_size=0.3, random_state= 22)
model_imp = DecisionTreeClassifier(random_state=22)
model_imp.fit(X_compl_train, Y_unica_train)
importance = model_imp.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()
np.where(importance>0.01)

X_train_rid = X_compl_train.iloc[:, [0,   2,   3,   5,   6,   8,   9,  10,  24,  32,  33,  34,  40,
         41,  42,  65,  79,  81,  82,  83,  84,  85,  86,  87,  89,  90,
         97,  99, 100]]
X_test_rid = X_compl_test.iloc[:, [0,   2,   3,   5,   6,   8,   9,  10,  24,  32,  33,  34,  40,
         41,  42,  65,  79,  81,  82,  83,  84,  85,  86,  87,  89,  90,
         97,  99, 100]]
class_tree=DecisionTreeClassifier(random_state=22)
param = {'max_depth':range(1, 6), 'min_samples_split':range(2,20), 'min_samples_leaf':range(2,15)} 
grid = GridSearchCV(class_tree, param, cv=3)
grid.fit(X_train_rid, Y_unica_train)
print(grid.best_params_)
mod_tree=DecisionTreeClassifier(max_depth=3,min_samples_split=19,min_samples_leaf=9,random_state=22)
mod_tree.fit(X_train_rid, Y_unica_train)
text_representation = tree.export_text(mod_tree)
print(text_representation)

fig1 = plt.figure(figsize=(25,20))
_ = tree.plot_tree(mod_tree,feature_names=list(X_train_rid.columns),filled=True)

y_train_pred = mod_tree.predict(X_train_rid)
print(classification_report(Y_unica_train, y_train_pred))
y_test_pred = mod_tree.predict(X_test_rid)
print(classification_report(Y_unica_test, y_test_pred))

#DECISION TREE WITH Y_UNICA -CLASS_WEIGHT=BALANCED
X_compl_train, X_compl_test, Y_unica_train, Y_unica_test = train_test_split(X_compl, Y_unica, test_size=0.3, random_state= 22)
model_imp = DecisionTreeClassifier(random_state=22, class_weight="balanced")
model_imp.fit(X_compl_train, Y_unica_train)
importance = model_imp.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))

plt.bar([x for x in range(len(importance))], importance)
plt.show()
np.where(importance>0.01)
X_train_rid = X_compl_train.iloc[:, [0,   2,   3,   5,   6,   8,  27,  32,  33,  41,  42,  47,  79,
         81,  82,  83,  84,  85,  86,  90,  91,  94,  95,  97, 103]]
X_test_rid = X_compl_test.iloc[:, [0,   2,   3,   5,   6,   8,  27,  32,  33,  41,  42,  47,  79,
         81,  82,  83,  84,  85,  86,  90,  91,  94,  95,  97, 103]]

class_tree=DecisionTreeClassifier(random_state=22, class_weight="balanced")
param = {'max_depth':range(1, 6), 'min_samples_split':range(2,20), 'min_samples_leaf':range(2,15)} 
grid = GridSearchCV(class_tree, param, cv=3)
grid.fit(X_train_rid, Y_unica_train)
print(grid.best_params_)
mod_tree=DecisionTreeClassifier(max_depth=5,min_samples_split=2,min_samples_leaf=2,random_state=22, class_weight="balanced")
mod_tree.fit(X_train_rid, Y_unica_train)
text_representation = tree.export_text(mod_tree)
print(text_representation)

fig1 = plt.figure(figsize=(25,20))
_ = tree.plot_tree(mod_tree,feature_names=list(X_train_rid.columns),filled=True)

y_train_pred = mod_tree.predict(X_train_rid)
print(classification_report(Y_unica_train, y_train_pred))
y_test_pred = mod_tree.predict(X_test_rid)
print(classification_report(Y_unica_test, y_test_pred))
