import streamlit as st
import pandas as pd
import joblib

st.title(" House Price Prediction")
st.write("""
Upload a CSV file with house features to predict SalePrice
""")

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    return model

model = load_model()

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("First 5 rows")
    st.dataframe(data.head())

    if "Id" not in data.columns:
        st.error("The CSV must have an 'Id' column.")
    else:
        ids = data["Id"]
        X = data.drop(columns=["Id"])

        preds = model.predict(X)

        st.write("Predictions")
        results = pd.DataFrame({"Id": ids, "Predicted_SalePrice": preds})
        st.table(results)


        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download submission.csv",
            data=csv,
            file_name='submission.csv',
            mime='text/csv'
        )
