import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import joblib

def datapreprocess(inputpath,outputpath,modelpath):
    
    # Checking all FilePaths:

    if not os.path.exists(inputpath):
        raise FileNotFoundError(f"file not found at {inputpath}")
    if not os.path.exists(outputpath):
        raise FileNotFoundError(f"file not found at {outputpath}")
    if not os.path.exists(modelpath):
        raise FileNotFoundError(f"file not found at {modelpath}")
    
                            #Data_preprocessing

    #importing csv
    df = pd.read_csv(inputpath)

    #crucial steps
    df.drop('customerID',axis=1,inplace=True)
    df['TotalCharges']=df['TotalCharges'].replace(' ',"0")
    df.TotalCharges = df.TotalCharges.astype(float)

    #converting target column without Encoding
    df['Churn'] = df["Churn"].replace({'Yes':1,'No':0})
    df.Churn = df.Churn.astype(int)

    #converting all other categorical columns with OneHotEncoder
    cat_col = df.select_dtypes(include='object').columns.to_list()
    labelEn = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    )
    arr = labelEn.fit_transform(df[cat_col])

    #adding the outputout array in the main dataframe
    fr = pd.DataFrame(arr,columns=labelEn.get_feature_names_out(cat_col))
    df.drop(cat_col,axis=1,inplace=True)
    df = pd.concat([df,fr],axis=1)

    #adding file to csv
    df.to_csv(outputpath+"/customer.csv",index=False)
    print("DONE!!!!!!")



    

if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    inputpath = os.path.join(BASE_DIR, "..", "data", "raw", "customer_raw.csv")
    outputpath = os.path.join(BASE_DIR, "..","data","processed")
    modelpath = os.path.join(BASE_DIR,"..","models","encoder.pkl")

    datapreprocess(inputpath,outputpath,modelpath)