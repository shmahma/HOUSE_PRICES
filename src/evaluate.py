import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

def evaluate(path, model_in="model.pkl", submission=False):
    model = joblib.load(model_in)

    df = pd.read_csv(path)

    if submission:
        ids = df["Id"]
        X = df.drop(columns=["Id"])
        preds = model.predict(X)
        submission_df = pd.DataFrame({"Id": ids, "SalePrice": preds})
        submission_df.to_csv("submission.csv", index=False)
        print("Fichier submission.csv généré !")

    else:
        X = df.drop(columns=["Id", "SalePrice"])
        y = df["SalePrice"]
        preds = model.predict(X)
        print("Évaluation locale :")
        print(f"R² : {r2_score(y, preds):.3f}")
        print(f"MAE : {mean_absolute_error(y, preds):.2f}")
        print(f"RMSE : {np.sqrt(mean_squared_error(y, preds)):.2f}")

if __name__ == "__main__":
    #evaluate("data/train.csv", submission=False)

    evaluate("data/test.csv", submission=True)
