#DATA PREPARATION
X_compl_train, X_compl_test, Y_unica_train, Y_unica_test = train_test_split(X_compl, Y_unica, test_size=0.3, random_state= 22)
Y_unica_train.value_counts()

strategy = {0:387, 9:121, 11:70, 10:50, 1:50, 8:50, 6:50, 4:45, 7:45, 3:40, 5:40, 2:40}
oversample = SMOTE(k_neighbors=2, random_state=22, sampling_strategy=strategy)
X_Smote_train, Y_Smote_train = oversample.fit_resample(X_compl_train,Y_unica_train)

counter = Counter(Y_Smote_train)
for k,v in counter.items():
	per = v / len(Y_Smote_train) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
plt.bar(counter.keys(), counter.values())
plt.show()
X_Smote_train=round(pd.DataFrame(X_Smote_train, columns=X_compl.columns),0); X_Smote_train
Y_Smote_train=pd.DataFrame(Y_Smote_train)
