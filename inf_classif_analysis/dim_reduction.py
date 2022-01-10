#AFTER DESCRIPTIVE ANALYSIS
#PCA AND T-SNE ON X_COMPL
X_compl_stand = StandardScaler().fit_transform(X_compl)
pca2= PCA(n_components=2, random_state=22)
pca_fit2 = pca2.fit_transform(X_compl_stand)
print(pca2.explained_variance_ratio_)
df=pd.DataFrame({'var':pca2.explained_variance_, 'PC':['PC1','PC2']})
sns.barplot(x='PC', y='var', data=df, color='c')
principalDf=pd.DataFrame(data=pca_fit2, columns=['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, Y_unica], axis = 1)
finalDf.head()
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)
targets = [0,1,2,3,4,5,6,7,8,9,10,11]
colors = ['red', 'green', 'black','purple','cyan','magenta','orange','blue','brown','yellow','grey','pink']
for target, color in zip(targets,colors):
    indicesToKeep=finalDf['Y_unica']==target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

pca = PCA(random_state=22)
pca_fit = pca.fit_transform(X_compl_stand)
pct_var_spiegata = pca.explained_variance_ / np.sum(pca.explained_variance_)
var_cum_spiegata = np.cumsum(pct_var_spiegata)
plt.plot(var_cum_spiegata)
plt.xlabel("N. componenti")
plt.ylabel("Var spiegata")
var_cum_spiegata
np.where(var_cum_spiegata>0.7)
#44 PRINCIPAL COMPONENTS--> EXPLAINED VARIANCE=70% 
#T-SNE ON THE 44 PCs
pca_44 = pd.DataFrame(pca_fit[:, 0:44])
tsne = TSNE(random_state=22)
tsne_fit = tsne.fit_transform(pca_44)
labels_Y_unica=Y_compl["Y_unica"]
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
 
targets = [0,1,2,3,4,5,6,7,8,9,10,11]
colors = ['red', 'green', 'black','purple','cyan','magenta','orange','blue','brown','yellow','grey','pink']
 
for target, color in zip(targets,colors):
    indicesToKeep=labels_Y_unica==target
    ax.scatter(tsne_fit[indicesToKeep, 0]
               , tsne_fit[indicesToKeep, 1]
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

#T-SNE ON ALL THE VARIABLES FROM X_COMPL
tsne = TSNE(random_state=22)
tsne_fit_all= tsne.fit_transform(X_compl_stand)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
 
targets = [0,1,2,3,4,5,6,7,8,9,10,11]
colors = ['red', 'green', 'black','purple','cyan','magenta','orange','blue','brown','yellow','grey','pink']
 
for target, color in zip(targets,colors):
    indicesToKeep=labels_Y_unica==target
    ax.scatter(tsne_fit_all[indicesToKeep, 0]
               , tsne_fit_all[indicesToKeep, 1]
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

#PCA AND T-SNE ON X
X_stand = StandardScaler().fit_transform(X)
pca2= PCA(n_components=2, random_state=22)
pca_fit2 = pca2.fit_transform(X_stand)
print(pca2.explained_variance_ratio_)
df=pd.DataFrame({'var':pca2.explained_variance_, 'PC':['PC1','PC2']})
sns.barplot(x='PC', y='var', data=df, color='c')
principalDf=pd.DataFrame(data=pca_fit2, columns=['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, Survive], axis = 1)
finalDf.head()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)
targets = [0,1]
colors = ['magenta', 'c']
for target, color in zip(targets,colors):
    indicesToKeep=finalDf['Survive']==target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
pca = PCA(random_state=22)
pca_fit = pca.fit_transform(X_stand)
pct_var_spiegata = pca.explained_variance_ / np.sum(pca.explained_variance_)
var_cum_spiegata = np.cumsum(pct_var_spiegata)
plt.plot(var_cum_spiegata)
plt.xlabel("N. componenti")
plt.ylabel("Var spiegata")
var_cum_spiegata
np.where(var_cum_spiegata>0.7)
#46 PCs TO HAVE 70% OF EXPLAINED VARIANCE
#T-SNE ON THESE 46 PCs
pca_46 = pd.DataFrame(pca_fit[:, 0:46])
tsne = TSNE(random_state=22)
tsne_fit = tsne.fit_transform(pca_46)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
 
targets = [0,1]
colors = ['magenta', 'c']
 
for target, color in zip(targets,colors):
    indicesToKeep=finalDf['Survive']==target
    ax.scatter(tsne_fit[indicesToKeep, 0]
               , tsne_fit[indicesToKeep, 1]
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

#T-SNE ALL THE VARIABLES OF X
tsne = TSNE(random_state=22)
tsne_fit_all= tsne.fit_transform(X_stand)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
 
targets = [0,1]
colors = ['magenta', 'c']
 
for target, color in zip(targets,colors):
    indicesToKeep=finalDf['Survive']==target
    ax.scatter(tsne_fit_all[indicesToKeep, 0]
               , tsne_fit_all[indicesToKeep, 1]
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
