X_train, X_test, Y_survive_train, Y_survive_test = train_test_split(X, survive, test_size=0.3, random_state= 22)
Logmodel = LogisticRegression(random_state=22, class_weight="balanced", n_jobs=-1)
Logmodel.fit(X_train, Y_survive_train)
importance = Logmodel.coef_[0]
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
plt.bar([x for x in range(len(importance))], importance)
plt.show()
np.where(abs(importance)>0.15)

X_train_rid = X_train.iloc[:, [1,   2,   4,   5,   6,   9,  24,  25,  28,  29,  36,  40,  41,
         42,  44,  45,  46,  49,  70,  82,  83,  86,  90,  91,  96,  98,
         99, 100, 103]]
X_test_rid = X_test.iloc[:, [1,   2,   4,   5,   6,   9,  24,  25,  28,  29,  36,  40,  41,
         42,  44,  45,  46,  49,  70,  82,  83,  86,  90,  91,  96,  98,
         99, 100, 103]]

class_log=LogisticRegression(random_state=22, max_iter= 200,solver="liblinear", class_weight="balanced")
param = dict(C=[1,2,3,4], penalty=["l1", "l2"])
grid = GridSearchCV(class_log, param, cv=5)
grid.fit(X_train_rid, Y_survive_train.values.ravel())
print(grid.best_params_)

class_log=LogisticRegression(C=3,max_iter=200, solver="liblinear", penalty="l1", random_state=22, class_weight="balanced")
class_log.fit(X_train_rid, Y_survive_train)
pred = class_log.predict(X_test_rid)
yhat_train= class_log.predict(X_train_rid)
print(classification_report(Y_survive_train, yhat_train))

pred_proba_test = class_log.predict_proba(X_test_rid)
pred_proba_train = class_log.predict_proba(X_train_rid)

skplt.metrics.plot_roc_curve(Y_survive_train, pred_proba_train, curves="each_class", cmap="brg")
plt.show()
skplt.metrics.plot_roc_curve(Y_survive_test, pred_proba_test, curves="each_class", cmap="brg")
plt.show()
