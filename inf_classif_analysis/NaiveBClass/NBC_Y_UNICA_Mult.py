MNaive=MultinomialNB()

#NBC WITH ALL THE VARIABLES
MNaive.fit(X_compl_train,Y_unica_train.values.ravel())
y_train_pred = MNaive.predict(X_compl_train)
print(classification_report(Y_unica_train, y_train_pred))
y_test_pred = MNaive.predict(X_compl_test)
print(classification_report(Y_unica_test, y_test_pred))

#NBC WITH FEATURE SELECTION
fs = SelectKBest(score_func = mutual_info_classif)
fs.fit(X_compl_train, Y_unica_train)
for i in range(len(fs.scores_)):
  print("Feature %d: %f" % (i, fs.scores_[i]))
  
 np.where(fs.scores_>0.020)
X_train_rid = X_compl_train.iloc[:, [  1,   2,   3,   4,   6,   8,   9,  18,  21,  24,  25,  26,  29,
         30,  32,  33,  45,  47,  50,  55,  59,  63,  65,  71,  79,  89,
         90,  94,  97, 100, 101]]
X_test_rid = X_compl_test.iloc[:, [ 1,   2,   3,   4,   6,   8,   9,  18,  21,  24,  25,  26,  29,
         30,  32,  33,  45,  47,  50,  55,  59,  63,  65,  71,  79,  89,
         90,  94,  97, 100, 101]]

MNaive.fit(X_train_rid,Y_unica_train.values.ravel())
y_train_pred = MNaive.predict(X_train_rid)
print(classification_report(Y_unica_train, y_train_pred))
y_test_pred = MNaive.predict(X_test_rid)
print(classification_report(Y_unica_test, y_test_pred))
