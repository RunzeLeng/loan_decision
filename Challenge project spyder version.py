import sqlite3
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif, RFE
from matplotlib import pyplot
from sklearn.linear_model import Lasso, Ridge
from sklearn.decomposition import FactorAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from numpy import loadtxt
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.tree import DecisionTreeClassifier



# ############################
# Build a single dataset for multiple sources
# ############################################

# creating file path
dbfile = 'C:/Users/rnlen/Desktop/ml_assessment.db'
# Create a SQL connection to our SQLite database
con = sqlite3.connect(dbfile)
# creating cursor
cur = con.cursor()


# reading all table names
table_list = [a for a in cur.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
# here is you table list
print(table_list)
#[('loan_information',), ('enterprise_information',), ('scores_information',), ('disbursement_information',)]


df_loan_information = pd.read_sql_query("SELECT * from loan_information", con)
df_enterprise_information = pd.read_sql_query("SELECT * from enterprise_information", con)
df_scores_information = pd.read_sql_query("SELECT * from scores_information", con)
df_disbursement_information = pd.read_sql_query("SELECT * from disbursement_information", con)
# Be sure to close the connection
con.close()


df1 = pd.merge(df_enterprise_information, df_scores_information, how='outer', on='loan_id')
df2 = pd.merge(df1, df_loan_information, how='outer', on='loan_id')

df2.describe()
df2 = df2.drop(df2.columns[[10]], axis=1)


df2['account_number']
df2['account_number'].value_counts()
df2['account_number'][0]
df2['account_number'][1]


# a = df2['account_number'].astype('bool')
# print(a)
# #df_withoutnull = df2.dropna(subset=['account_number'])
# #df_withoutnull55 = df2.loc[-a]
# df2_without_none = df2[a]
# df2_with_none = df2[-a]


# df_merge_without_none = pd.merge(df2_without_none, df_disbursement_information, how='left', on='account_number')
# df_merge_without_none.shape
# df_merge_without_none['total_disbursement_amount'].describe()
# df_merge_without_none['total_disbursement_amount'].median()
# total_disbursement_amount = pd.Series(300000, index =range(0,1107))


a = df_disbursement_information['account_number'].astype('bool')
print(a)
#df_withoutnull = df2.dropna(subset=['account_number'])
#df_withoutnull55 = df2.loc[-a]
df_disbursement_information_without_none = df_disbursement_information[a]


df_merge = pd.merge(df2, df_disbursement_information_without_none, how='left', on='account_number')
df_merge.shape
df_merge.head(10)
df_merge.describe()


# #####################
# Clean the dataset
# #####################
df = df_merge.drop(columns=['loan_id', 'customer_id','enterprise_id_x','hub_id'])


df['disbursement_month'].isna().sum() #1107
df['screening_date'].isna().sum() #126
df = df.drop(columns=['disbursement_month'])


### Drop NA
df.isna().sum()
df = df.dropna(subset=['business_activity','business_sector','ManagAgeui_APP','ManagCBscoreui_APP','screening_date'])
df.isna().sum()



df['business_type'].value_counts()
df['BusinFormalityOfTheBusinessui'].value_counts()
df['account_number'].value_counts()
df['product_code'].value_counts()
df['total_disbursement_amount'].value_counts()


df.info()
df['business_type'] = df['business_type'].fillna("Unknown")
df['business_type'].value_counts()


df['BusinFormalityOfTheBusinessui'] = df['BusinFormalityOfTheBusinessui'].fillna("Unknown")
df['BusinFormalityOfTheBusinessui'].value_counts()


df['total_disbursement_amount'] = df['total_disbursement_amount'].fillna(df['total_disbursement_amount'].median())
df['total_disbursement_amount'].value_counts()



df['product_code'] = df['product_code'].replace(regex={r'^....S$': 'S', r'^....U$': 'U'})
df['product_code'].value_counts()
df['product_code'] = df['product_code'].fillna("Unknown")
# imputer = SimpleImputer(missing_values=None, strategy='constant', fill_value='X')    ## S before
# df.product_code = imputer.fit_transform(df['product_code'].values.reshape(-1,1))[:,0]
df['product_code'].value_counts()


a = df['account_number'].astype('bool')
df['account_number'] = a.astype('category')
df['account_number'].value_counts()
df.info()
df.isna().sum()


df1 = pd.get_dummies(df, columns=["business_type", "business_activity"], prefix=["btype", "bactivity"])
df1.info()
df2 = pd.get_dummies(df1, columns=["business_sector", "BusinFormalityOfTheBusinessui"], prefix=["bsector", "bFormality"])
df2.info()
df3 = pd.get_dummies(df2, columns=["account_number", "product_code"], prefix=["account", "product"])
df3.info()


df3['year'] = df3['screening_date'].replace(regex={r'^2008......$': '2008', r'^2014......$$': '2014', r'^2016......$$': '2016', 
                                                 r'^2017......$$': '2017', r'^2018......$$': '2018', r'^2019......$$': '2019'})
df3['month'] = df3['screening_date'].replace(regex={r'^.....01...$': '01', r'^.....02...$': '02', r'^.....03...$': '03', r'^.....04...$': '04',
                                                 r'^.....05...$': '05', r'^.....06...$': '06', r'^.....07...$': '07', r'^.....08...$': '08',
                                                 r'^.....09...$': '09', r'^.....10...$': '10', r'^.....11...$': '11', r'^.....12...$': '12'})
df3 = df3.drop(columns=['screening_date'])
df3.info()


df4 = pd.get_dummies(df3, columns=["year", "month"], prefix=["year", "month"])
df4.info()


df4['ManagCBscoreui_APP'] = np.where(df4['ManagCBscoreui_APP'] < 700, 1, 0) ##<700 1 reject     ------        >=700 0 approve
data = df4.rename(columns={'ManagCBscoreui_APP': 'loandecision'})
data.info()


df['ManagCBscoreui_APP'] = np.where(df['ManagCBscoreui_APP'] < 700, 1, 0) ##<700 1 reject     ------        >=700 0 approve
df = df.rename(columns={'ManagCBscoreui_APP': 'loandecision'})
df.info()
                                        




# #################################################
# Perform EDA
# ################################################
data = data.drop(columns=data.columns[61:]) ## 61 need to change when data columns change
data = data.reset_index(drop=True)
y = data['loandecision']
x = data.drop(columns=['loandecision'])




############################################################### Assemble new dataset
############################
x_int = x[x.columns[0:2]] #(2348,2) int
x_noint = x.drop(columns=x.columns[0:2]) #(2348,57) category
x = x_noint #(2348,57)

# fs = SelectKBest(chi2, k=10)
# x_new = fs.fit_transform(x, y) #(2348,k=5)

x = pd.DataFrame(x_new) #(2348,k)
x = pd.concat([x, x_int], axis=1) #(2348,k+2) combined dataset
x = x.rename(str, axis='columns')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0) #split

for i in range(len(fs.scores_)):
 	print('%s: %f' % (x_noint.columns[i], fs.scores_[i]))
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()

################################################################ Dimension Reduction
################1
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


################2
fs = SelectKBest(chi2, k=3)
x_new = fs.fit_transform(x, y)
x = pd.DataFrame(x_new).rename(str, axis='columns')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


################3
fs = SelectKBest(mutual_info_classif, k=15)
x_new = fs.fit_transform(x, y)
x = pd.DataFrame(x_new).rename(str, axis='columns')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


################4
fs = SelectKBest(f_classif, k=3)  ##ANOVA
x_new = fs.fit_transform(x, y)
x = pd.DataFrame(x_new).rename(str, axis='columns')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)



#################5   Multiple factor analysis
fs = FactorAnalysis(n_components=10, random_state=0)
x_new = fs.fit_transform(x)
x = pd.DataFrame(x_new).rename(str, axis='columns')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)



#############################################################################  Recursive Feature Elimination (RFE) for Feature Selection
################################        REF
rfe = RFE(estimator = DecisionTreeClassifier() , n_features_to_select=10)
rfe = RFE(estimator = RandomForestClassifier(n_estimators=100, max_depth=4) , n_features_to_select=10)
rfe = RFE(estimator = Lasso(alpha=1e-4) , n_features_to_select=10)
rfe = RFE(estimator = LogisticRegression() , n_features_to_select=10)


fit = rfe.fit(x_train, y_train)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
y_pred = fit.predict(x_test)

accuracy_score(y_test, y_pred)
accuracy_score(y_test, (y_pred > 0.5).astype(int))  ## 0.5 is Temporary




########################################################################## Classification Model
#################### Lasso
alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3, 1e-2, 1, 5, 10]
b = np.arange(0.35, 0.65, 0.01).tolist()
F = []
T = []

for j in range(10):
    lasso = Lasso(alpha=alpha_lasso[j])
    lasso.fit(x_train,y_train)
    y_pred = lasso.predict(x_test)
    y_trainpred = lasso.predict(x_train)
    
    for i in range(len(b)):
        yp = (y_pred > b[i]).astype(int)
        tp = (y_trainpred > b[i]).astype(int)
        F.append(accuracy_score(y_test, yp))
        T.append(accuracy_score(y_train, tp))
        print(confusion_matrix(y_test, yp))
        
plt.plot(T,'bx')
plt.plot(F,'rx')
plt.show()


#################################### LOGISTIC
b = np.arange(0.35, 0.65, 0.01).tolist()
F = []
T = []

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
print(accuracy_score(y_test, y_pred))

for i in range(len(b)):
    yp = (logreg.predict_proba(x_test)[:,1] > b[i]).astype(int)
    tp = (logreg.predict_proba(x_train)[:,1] > b[i]).astype(int)
    F.append(accuracy_score(y_test, yp))
    T.append(accuracy_score(y_train, tp))
    print(confusion_matrix(y_test, yp))
    
plt.plot(T,'bx')
plt.plot(F,'rx')
plt.show()

# logreg.coef_
# logreg.intercept_
# logreg.predict_proba(x_test)
# np.quantile(c, 0.6)

######################### kernel SVM
b = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
F = []
T = []

for i in range(13):
    for j in range(13):
        classifier = SVC(kernel = 'rbf', C = b[i], gamma = b[j])
        classifier.fit(x_train, y_train)
        yp = classifier.predict(x_test)
        tp = classifier.predict(x_train)
        F.append(accuracy_score(y_test, yp))
        T.append(accuracy_score(y_train, tp))
        print(confusion_matrix(y_test, yp))
        
plt.plot(T,'bx')
plt.plot(F,'rx')
plt.show()



################################################################################ Deep learning
##################################
length = len(x.columns)
def NN_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=length, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(40, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = NN_model()
model.summary()

# early_stop = EarlyStopping(patience=3, monitor='val_loss')   ## Regulation 
model.fit(x_train, y_train, batch_size=50, validation_data=(x_test, y_test), epochs=50, verbose=1) ## callbacks=[early_stop]

scores = model.evaluate(x_test, y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))  

yp = model.predict(x_test) ## numerical between 0 and 1

    


########################################### Random Forest
clf = RandomForestClassifier(n_estimators=100)
clf.fit(x_train,y_train)

yp = clf.predict(x_test)
accuracy_score(y_test, yp)



#############################################################################   CROSS VALIDATION
##################################
clf = RandomForestClassifier(n_estimators=100, max_depth=5)  ## max_depth=3
scores = cross_validate(clf, x_train, y_train, cv=2, scoring='accuracy', return_train_score=True, return_estimator=True)

T = scores['train_score']
F = scores['test_score']
plt.plot(T,'bx')
plt.plot(F,'rx')
plt.show()

# model = scores['estimator'][1]
# yp = model.predict(x_test)
# accuracy_score(y_test, yp)

# ytp = model.predict(x_train)
# accuracy_score(y_train, ytp)

