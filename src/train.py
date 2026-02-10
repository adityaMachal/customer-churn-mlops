import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def ModelTrain(filepath,outpath):
    #accessing processed data
    df = pd.read_csv(filepath)

    #seperating Training and Target Column
    X = df.drop('Churn',axis=1)
    y = df['Churn']

    #Handling Imbalance Dataset for unbaised prediction
    sm = SMOTE(random_state=42)
    Xsm,Ysm = sm.fit_resample(X,y)

    #Training a RandomForest Model
    model = RandomForestClassifier(
        max_leaf_nodes=60,
        min_samples_split=2,
        n_estimators=100,
        class_weight=None,
        max_depth=None
        )
    model.fit(X,y)

    #Saving model


    joblib.dump(model,outpath)
    print(f"Model saved in {outpath}")
    
    pass
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(BASE_DIR,"..","data","processed","customer.csv")
    outpath = os.path.join(BASE_DIR,'..','models','RF.pkl')
    ModelTrain(filepath,outpath)