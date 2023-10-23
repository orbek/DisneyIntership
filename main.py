import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def main():
    st.title("Client Purchase Prediction")

    st.write("""
This tool is designed to train a machine learning model using historical data to forecast whether a client will make a purchase in the upcoming month. It aims to generate a list of clients predicted not to buy (indicated by a prediction of 0). Marketing efforts can then be focused on this list to enhance revenue.
    """)

    # Upload data files
    data_file = st.file_uploader("Upload Client History data", type=["csv"])
    if data_file:
        data = pd.read_csv(data_file).set_index('client_id')
        st.write("This dataset captures records from 100,000 clients, detailing their age, annual income, and purchase history spanning from September 2022 to August 2023. It tracks both the products bought and the total number of orders placed by each client. Additionally, there's a column named 'purchase_next_month' which indicates whether a client placed an order in September 2023.")
        # Let the user select the columns to use as features
        features = st.multiselect('Select Features:', data.columns)
        # Let the user select the target column
        default_target = 'purchase_next_month'
        columns_for_target = [default_target] + [col for col in data.columns if col != default_target]
        target = st.selectbox('Select Target Column: (What you want to predict)', columns_for_target)

        data = data[features + [target]]
        
        st.write("### Data Overview")
        st.write(data.head())

        st.write("### Features Correlation")
        correlation = data[features].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        try:
            sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=ax)
            st.pyplot(fig)
        except ValueError:
            st.warning("Unable to plot heatmap due to empty or invalid data.")

        
        # Hyperparameters
        if len(features) == 0:
            st.warning("Please select the features and target to proceed.")
        else:
            max_pca_components = len(features)
            n_components = st.slider('Number of PCA Components', min_value=1, max_value=max_pca_components, value=min(7, max_pca_components), step=1)
        C = st.slider('Inverse of Regularization Strength (C)', min_value=0.001, max_value=10.0, value=0.01, step=0.001)
        max_iter = st.slider('Max Iterations', min_value=100, max_value=1000, value=500, step=100)
        test_size = st.slider('Test Set Size (%)', min_value=10, max_value=90, value=20, step=5)
        
        st.write(f"Distribution of {target}:")
        st.write(data[target].value_counts())
        
        # Scaling the data
        scaler = StandardScaler()
        data_to_scale = data.drop(columns=[target])
        if data_to_scale.empty:
            st.warning("Please select at least one feature to proceed.")
            return
        else:
            scaled_data = scaler.fit_transform(data_to_scale)

        try:
            # PCA
            pca = PCA(n_components=n_components)
            principalcomponents = pca.fit_transform(scaled_data)
            explained_variance = pca.explained_variance_ratio_

            st.write(f"Explained Variance for {n_components} components:", explained_variance.sum())
        except Exception as e:
            st.error(f"An error occurred during model training or prediction: {str(e)}. Please check your data and try again.")

        # Splitting the data
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(principalcomponents, y, test_size=test_size/100, stratify=y)

        # Logistic Regression
        logistic_model = LogisticRegression(C=C, max_iter=max_iter, solver='liblinear')
        logistic_model.fit(X_train, y_train)

        y_pred = logistic_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Display results
        st.write("#### Results")
        st.write(f"Test Accuracy: {accuracy:.4f}")

        st.write("### Predictions")
        # Upload data to be predicted
        predict_data_file = st.file_uploader("Upload Client History data for predictions", type=["csv"])
        if predict_data_file:
            new_data = pd.read_csv(predict_data_file).set_index('client_id')
            
            # Only select the columns the model is trained on
            X_new = new_data[features]
            X_new_scaled = scaler.transform(X_new)
            X_new_pca = pca.transform(X_new_scaled)
            predictions = logistic_model.predict(X_new_pca)

            # Adding predictions to the new dataset for reference
            new_data['predicted_purchase_next_month'] = predictions

            # Separate the new dataset based on predictions
            clients_with_pred_0 = new_data[new_data['predicted_purchase_next_month'] == 0].index
            clients_with_pred_1 = new_data[new_data['predicted_purchase_next_month'] == 1].index

            # Display predictions overview
            st.write("### Predictions Overview")
            st.write(new_data.head())

            # Display summary
            value_counts = new_data['predicted_purchase_next_month'].value_counts()
            st.write("### Summary")
            st.write("Value Counts:\n", value_counts)

            # Provide option to download datasets
            st.write("### Download Client Lists based on Predictions")
            if st.button("Download clients predicted as 0"):
                st.download_button("Clients_Predicted_0.csv", pd.DataFrame(clients_with_pred_0).to_csv(index=False))
            if st.button("Download clients predicted as 1"):
                st.download_button("Clients_Predicted_1.csv", pd.DataFrame(clients_with_pred_1).to_csv(index=False))


if __name__ == "__main__":
    main()