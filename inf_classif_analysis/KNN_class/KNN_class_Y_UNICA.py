X_compl_train, X_compl_test, Y_unica_train, Y_unica_test = train_test_split(X_compl, Y_unica, test_size=0.3, random_state= 22)
fs = SelectKBest(score_func = mutual_info_classif)
fs.fit(X_compl_train, Y_unica_train)
for i in range(len(fs.scores_)):
  print("Feature %d: %f" % (i, fs.scores_[i]))
  np.where(fs.scores_>0.015)
  
  X_train_rid=X_compl_train.iloc[:,[1,   3,   4,   5,   6,   7,   8,   9,  14,  16,  17,  20,  29,
         32,  33,  35,  43,  45,  46,  49,  50,  55,  56,  60,  72,  73,
         78,  86,  90,  91,  97,  98,  99, 100, 101, 102]]
  X_test_rid=X_compl_test.iloc[:,[1,   3,   4,   5,   6,   7,   8,   9,  14,  16,  17,  20,  29,
         32,  33,  35,  43,  45,  46,  49,  50,  55,  56,  60,  72,  73,
         78,  86,  90,  91,  97,  98,  99, 100, 101, 102]]
ss=StandardScaler()
X_stand_train = ss.fit_transform(X_train_rid)
X_stand_test= ss.transform(X_test_rid)
knn=KNeighborsClassifier()

grid_params=dict(n_neighbors=[3,5,7,9,11], weights=["uniform","distance"],metric=["euclidean","manhattan"])
gs_KNN= GridSearchCV(knn, grid_params, n_jobs=-1, cv=3, verbose=1)
gs_KNN.fit(X_stand_train,Y_unica_train.values.ravel())
print(gs_KNN.best_params_)

knn=KNeighborsClassifier(metric="euclidean",n_neighbors=7, weights="uniform")
knn.fit(X_stand_train,Y_unica_train)
y_train_pred = knn.predict(X_stand_train)
print(classification_report(Y_unica_train, y_train_pred))
y_test_pred = knn.predict(X_stand_test)
print(classification_report(Y_unica_test, y_test_pred))
