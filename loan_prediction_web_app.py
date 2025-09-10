import numpy as np
import pickle
import streamlit as st

## Loading the saved model - right click on the file name on the left here in VSCode and click on path. 
# Paste the copied path below for the file to be opened.
loaded_model = pickle.load(open("loan_prediction_model_v2.pkl", "rb")) # rb means read binary

def loan_prediction(input_data):
        
    # Changing the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array so the model will understand I am making a prediction for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Making prediction on the loaded model
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return("This customer is not eligible for a loan")
    else:
        return("This customer is eligible for a loan")
    
def main():

    # Giving the app a title
    st.title("Loan Prediction Web App")
            
    # Getting the input data from user
    gender = st.text_input("gender")
    married = st.text_input("married")
    education = st.text_input("education")
    self_employed = st.text_input("self_employed")
    applicant_income = st.text_input("applicant_income")
    coapplicant_income = st.text_input("coapplicant_income")
    loan_amount = st.text_input("loan_amount")
    loan_amount_term = st.text_input("loan_amount_term")
    dependents_1 = st.text_input("dependents_1")
    dependents_2 = st.text_input("dependents_2")
    dependents_3 = st.text_input("dependents_3+")
    credit_history = st.text_input("credit_history")
    property_area_Semiurban = st.text_input("property_area_Semiurban")
    property_area_Urban = st.text_input("property_area_Urban")

    # Code for performance
    performace = ""

    # Creating a button for prediction

    if st.button("Loan status"):
        performance = loan_prediction(
            [int(gender), int(married), int(education),int(self_employed), int(applicant_income), int(coapplicant_income), int(loan_amount), 
                int(loan_amount_term), int(dependents_1), int(dependents_2), int(dependents_3), int(credit_history), int(property_area_Semiurban), 
                int(property_area_Urban)])
        
        st.success(performance)

if __name__ =="__main__":
    main()
