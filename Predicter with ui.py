import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flaml import AutoML
import lightgbm as lgb
from imblearn.over_sampling import SMOTEN

st.title("PCOS Diagnosis")

automl = AutoML()
automl_settings = {
    "time_budget": 120,
    "metric": 'accuracy',
    "task": 'classification',
    "log_file_name": "pcos.log",
}

df_inf=pd.read_csv(r"C:\Users\STUDY.LAPTOP-R70IIG6F\PycharmProjects\PCOS\New folder\PCOS_infertility.csv")
df_noinf=pd.read_excel(r"C:\Users\STUDY.LAPTOP-R70IIG6F\PycharmProjects\PCOS\New folder\PCOS_data_without_infertility (5).xlsx",sheet_name="Full_new")

df=pd.merge(df_noinf,df_inf, on='Patient File No.',suffixes={'','_y'},how='left')
df=df.drop(['Unnamed: 44', 'Sl. No_y', 'PCOS (Y/N)_y', '  I   beta-HCG(mIU/mL)_y','II    beta-HCG(mIU/mL)_y', 'AMH(ng/mL)_y'], axis=1)
df.head()

df["AMH(ng/mL)"] = pd.to_numeric(df["AMH(ng/mL)"], errors='coerce')
df["II    beta-HCG(mIU/mL)"] = pd.to_numeric(df["II    beta-HCG(mIU/mL)"], errors='coerce')

df['Marraige Status (Yrs)'].fillna(df['Marraige Status (Yrs)'].median(),inplace=True)
df['II    beta-HCG(mIU/mL)'].fillna(df['II    beta-HCG(mIU/mL)'].median(),inplace=True)
df['AMH(ng/mL)'].fillna(df['AMH(ng/mL)'].median(),inplace=True)
df['Fast food (Y/N)'].fillna(df['Fast food (Y/N)'].median(),inplace=True)

df.columns = [col.strip() for col in df.columns]

df.to_csv("jemp.csv")

#model building
X=df.drop(["PCOS (Y/N)","Sl. No","Patient File No."],axis = 1)
y=df["PCOS (Y/N)"]

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
imp =importances.nlargest(10)
df2 = df[imp.index]
sm = SMOTEN(random_state = 42)
X_sm, y_sm = sm.fit_resample(df2, y)
X_sm=pd.DataFrame(X_sm, columns=df2.columns)
y_sm=pd.DataFrame(y_sm, columns=['PCOS (Y/N)'])
print('New balance of 1 and 0 classes (%):')
y_sm.value_counts()
X_train,X_test, y_train, y_test = train_test_split(X_sm,y_sm, test_size=0.3, random_state=12)
import numpy as np
y_train = np.array(y_train)
y_test = np.array(y_test)
# automl.fit(X_train=X_train, y_train=y_train,X_val=X_test, y_val=y_test,**automl_settings)
# print(automl.model.estimator)
rfc = lgb.LGBMClassifier(colsample_bytree=0.8055416592222705,
               learning_rate=0.31502410263124375, max_bin=511,
               min_child_samples=35, n_estimators=17, num_leaves=5,
               reg_alpha=0.03904733197824847, reg_lambda=2.3879996428323564,
               verbose=-1)
rfc.fit(X_train, y_train)
#Making prediction and checking the test set
pred_rfc = rfc.predict(X_test)
accuracy = accuracy_score(y_test, pred_rfc)
print(accuracy)
save_to = '{}.txt'.format('pcos_model')
rfc.booster_.save_model(save_to)
Follicle_No_R = st.number_input("Enter Follicle No. (R) : ")
Skin_darkening = st.number_input("Enter Skin darkening (Y/N) : ")
Follicle_No_L = st.number_input("Enter Follicle No. (L) : ")
hair_growth = st.number_input("Enter Hair growth (Y/N) : ")
Weight_gain = st.number_input("Enter Weight gain (Y/N) : ")
Cycle = st.number_input("Enter Cycle(R/I) : ")
Pimples = st.number_input("Enter Pimples (Y/N) : ")
Fast_food = st.number_input("Enter Fast food (Y/N) : ")
Cycle_length = st.number_input("Enter Cycle length (days) : ")
AMH = st.number_input("Enter AMH (ng/mL) : ")
yes = rfc.predict([[Follicle_No_R,Skin_darkening,hair_growth,Follicle_No_L,Weight_gain,Cycle,Fast_food,Pimples,Cycle_length,AMH]])
if yes[0] == 1:
    st.write("Hai")
else:
    st.write("Nahi hai")