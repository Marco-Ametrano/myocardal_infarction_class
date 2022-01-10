!pip install scikit-plot
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import plotly.express as px
import keras
import scikitplot as skplt
import xgboost
import imblearn

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

from google.colab import files
dataset = files.upload()

dataset = pd.read_csv("Myocardial infarction complications Database.csv"); dataset

#PREPROCESSING
dataset.isnull().sum()

col=dataset.columns
colors=['green','red']
sns.heatmap(dataset[col].isnull(),cmap=sns.color_palette(colors))

for col in dataset.columns:
  pct_missing=np.mean(dataset[col].isnull())
  print('{} {}%'.format(col,round(pct_missing*100)))

dataset=dataset.drop(['IBS_NASL','KFK_BLOOD','NOT_NA_KB','S_AD_KBRIG','D_AD_KBRIG','LID_KB', 'NA_KB'],axis=1)
missing_row=dataset.isnull().sum(axis=1)
miss_counts = missing_row.value_counts(); miss_counts
sns.barplot(miss_counts.index, miss_counts.values, color= "steelblue")

dataset=dataset.dropna(axis=0,thresh=112)
dataset = dataset.reset_index(drop=True)
col=dataset.columns
for col in dataset.columns:
  missing= dataset[col].isnull().sum()
  print('{} {}'.format(col, missing))
subset_cat = dataset[['INF_ANAM', 'STENOK_AN', 'FK_STENOK', 'IBS_POST', 'GB', 'SIM_GIPERT', 'DLIT_AG', 'ZSN_A', 'endocr_01', 'endocr_02', 'endocr_03', 
                      'zab_leg_01', 'zab_leg_02', 'zab_leg_03', 'zab_leg_04', 'zab_leg_06', 'K_SH_POST', 'ant_im', 'lat_im', 'inf_im', 'post_im', 
                      'IM_PG_P', 'GIPO_K', 'GIPER_NA', 'TIME_B_S', 'R_AB_1_n', 'R_AB_2_n','R_AB_3_n', 'NA_R_3_n', 'NOT_NA_3_n', 'LID_S_n', 
                      'ANT_CA_S_n', 'GEPAR_S_n', 'ASP_S_n', 'TIKL_S_n', 'TRENT_S_n']]

imp_mode = SimpleImputer(strategy='most_frequent')
array_cat = imp_mode.fit_transform(subset_cat)
subset_cat_nonan = pd.DataFrame(array_cat, columns=subset_cat.columns)
subset_cont = dataset[['AGE', 'S_AD_ORIT', 'D_AD_ORIT', 'K_BLOOD', 'NA_BLOOD', 'ALT_BLOOD', 'AST_BLOOD', 'L_BLOOD', 'ROE']]
imp_mean = SimpleImputer(strategy='mean')
array_cont = imp_mean.fit_transform(subset_cont)
subset_cont_nonan = round(pd.DataFrame(array_cont, columns=subset_cont.columns), 2)
cat_cont = pd.concat([subset_cat_nonan, subset_cont_nonan], axis=1)

dataset_noimp = dataset.drop(['INF_ANAM', 'STENOK_AN', 'FK_STENOK', 'IBS_POST', 'GB', 'SIM_GIPERT', 'DLIT_AG', 'ZSN_A', 'endocr_01', 'endocr_02', 
                              'endocr_03', 'zab_leg_01', 'zab_leg_02', 'zab_leg_03', 'zab_leg_04', 'zab_leg_06', 'K_SH_POST', 'ant_im', 'lat_im', 
                              'inf_im', 'post_im','IM_PG_P', 'GIPO_K', 'GIPER_NA', 'TIME_B_S', 'R_AB_1_n', 'R_AB_2_n','R_AB_3_n', 'NA_R_3_n', 
                              'NOT_NA_3_n', 'LID_S_n', 'ANT_CA_S_n', 'GEPAR_S_n', 'ASP_S_n', 'TIKL_S_n', 'TRENT_S_n', 'AGE', 'S_AD_ORIT', 
                              'D_AD_ORIT', 'K_BLOOD', 'NA_BLOOD', 'ALT_BLOOD', 'AST_BLOOD', 'L_BLOOD', 'ROE'], axis=1)

newdataset = pd.concat([cat_cont, dataset_noimp], axis=1)
newdataset = newdataset[dataset.columns]; newdataset

col=newdataset.columns
newdataset[col].isnull().any()

dataset_input = newdataset.drop(newdataset.iloc[:, 105:117], axis=1)
X = dataset_input.drop("ID", axis=1)

#TARGET #VARIABLES #DEFINITION
Y_noLET_IS=newdataset.iloc[:, 105:116]
data=np.where(Y_noLET_IS.sum(axis=1)<=1)
datacompl=newdataset.iloc[data]
datacompl=datacompl.reset_index(drop=True)
X_compl=datacompl.drop(datacompl.iloc[:, 105:117], axis=1)
X_compl=X_compl.drop('ID',axis=1)
Y_compl=datacompl.iloc[:,105:116]
Y_compl['no_compl']=''
for i in range(0,len(Y_compl)):
  if Y_compl.iloc[i,0:11].sum()==0:
    Y_compl.iloc[i,11]=1
  else:
     Y_compl.iloc[i,11]=0
     Y_compl
Y_compl['Y_unica']=''
for i in range(0,len(Y_compl)):
  if Y_compl.iloc[i,0]==1:
    Y_compl.iloc[i,12]=1
  elif  Y_compl.iloc[i,1]==1:
    Y_compl.iloc[i, 12]=2
  elif  Y_compl.iloc[i,2]==1:
    Y_compl.iloc[i, 12]=3
  elif  Y_compl.iloc[i,3]==1:
    Y_compl.iloc[i, 12]=4
  elif  Y_compl.iloc[i,4]==1:
    Y_compl.iloc[i, 12]=5
  elif  Y_compl.iloc[i,5]==1:
    Y_compl.iloc[i, 12]=6
  elif  Y_compl.iloc[i,6]==1:
    Y_compl.iloc[i, 12]=7
  elif  Y_compl.iloc[i,7]==1:
    Y_compl.iloc[i, 12]=8
  elif  Y_compl.iloc[i,8]==1:
   Y_compl.iloc[i, 12]=9
  elif  Y_compl.iloc[i,9]==1:
    Y_compl.iloc[i, 12]=10
  elif  Y_compl.iloc[i,10]==1:
   Y_compl.iloc[i, 12]=11
  elif  Y_compl.iloc[i,11]==1:
    Y_compl.iloc[i, 12]=0

Y_unica=Y_compl[['Y_unica']]
Y_unica = Y_unica.astype("category")
LET_IS=pd.DataFrame(newdataset['LET_IS'])
LET_IS['Survive']=''
for i in range(0,len(LET_IS.index)):
  if LET_IS.iloc[i, 0]==0:
    LET_IS.iloc[i,1]=1
  else:
    LET_IS.iloc[i,1]=0;
LET_IS

Survive=LET_IS[["Survive"]]; Survive
LET_IS=LET_IS.drop(["Survive"],axis=1)
ZSN = Y_noLET_IS[["ZSN"]]; ZSN
