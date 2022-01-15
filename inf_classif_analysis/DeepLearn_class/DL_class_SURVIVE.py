X_train, X_test, Survive_train, Survive_test = train_test_split(X, Survive, test_size=0.3, random_state= 22)
npYtrain=np.asarray(Survive_train).astype('float32')
npXtrain=np.asarray(X_train).astype('float32')
npYtest=np.asarray(Survive_test).astype('float32')
npXtest=np.asarray(X_test).astype('float32')

model = Sequential()
model.add(Dense(32, input_dim=104, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(npXtrain, npYtrain , epochs=500, batch_size=22)

_, accuracy = model.evaluate(npXtrain, npYtrain)
print('Accuracy: %.2f' % (accuracy*100))
_, accuracy = model.evaluate(npXtest, npYtest)
print('Accuracy: %.2f' % (accuracy*100))

pred_train=pd.DataFrame(model.predict(npXtrain))
preds_train=round(pred_train)
preds_train.value_counts()
pred_test=pd.DataFrame(model.predict(npXtest))
preds_test=round(pred_test)
preds_test.value_counts()

print(confusion_matrix(npYtrain,preds_train))
print(confusion_matrix(npYtest,preds_test))
print(classification_report(npYtrain,preds_train))

print(classification_report(npYtest,preds_test))
