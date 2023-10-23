# DisneyIntership
## Project for Commercial DS Internship

This project was crafted as a component of the selection process for a Disney internship.

Utilizing the Streamlit framework on python, the application is designed to train a Logistic Regression machine learning model using historical data. Once trained, the model can identify and list clients predicted to abstain from making any purchases in the subsequent month. Armed with this list, the marketing team can strategize targeted campaigns to enhance revenue.


# Instructions:

1 - Install necessary dependencies with: `pip install -r requirements.txt`
2 - Launch the application using: `streamlit run main.py`
3 - Upload the `Clienthistory.csv` file to train the algorithm. Adjust hyperparameters as needed to fine-tune model accuracy.
4 - Upload the `Clienthistorypred.csv` file for predictions on the Purchase_next_month column. Predicted values are 0 (not buying) and 1 (buying).
5 - Export the processed data for further analysis or reporting.
