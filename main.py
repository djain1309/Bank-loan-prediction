import  plotly.express as px
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(layout="wide")
st.title('Bank Loan Status Prediction')

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

xgb_model = joblib.load("xgboost_pipeline.pkl")
label_encoder_dict = joblib.load("label_encoder_dict.pkl")
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.subheader("Select the Input Type")
col = st.sidebar.columns(1)

input_type = st.selectbox("", ("Input the variables", "Upload CSV"))

if input_type == "Input the variables":
    current_loan_amount = col[0].number_input("Current Loan Amount",
                                              min_value=0.0)
    term = col[0].selectbox("Term", ('Short Term', 'Long Term'))

    credit_score = col[0].number_input("Credit Score", min_value=300,
                                       max_value=850, value=650)

    annual_income = col[0].number_input("Annual Income", min_value=0.0,
                                        value=76627.0)

    years_in_current_job = col[0].number_input("Years in current job",
                                               min_value=0,
                                               max_value=15, value=3)

    home_ownership = col[0].selectbox("Home Ownership",
                                      ("Home Mortgage", "Rent",
                                       "Own Home", "HaveMortgage"))

    purpose_options = ('Home Improvements', 'Debt Consolidation', 'Buy House',
                       'other', 'Business Loan', 'Buy a Car', 'major_purchase',
                       'Take a Trip', 'Other', 'small_business',
                       'Medical Bills',
                       'wedding', 'vacation', 'Educational Expenses', 'moving',
                       'renewable_energy')

    purpose = col[0].selectbox("Purpose", purpose_options)

    monthly_debt = col[0].number_input("Monthly Debt", min_value=0.0)

    years_of_credit_history = col[0].number_input(
        "Years of Credit History", min_value=0.0, max_value=100.0, value=17.0
    )

    number_of_open_accounts = col[0].number_input(
        "Number of Open Accounts", min_value=0, max_value=100, value=20
    )

    number_of_credit_problems = col[0].number_input(
        "Number of Credit Problems", min_value=0
    )
    current_credit_balance = col[0].number_input("Current Credit Balance",
                                                 min_value=0.0)

    maximum_open_credit = col[0].number_input("Maximum Open Credit",
                                              min_value=0.0)

    bankruptcies = col[0].number_input("Bankruptcies", min_value=0, value=0)

    tax_liens = col[0].number_input("Tax Liens", min_value=0)

    pr = col[0].button("Predict")
    if pr:
        loan_status = xgb_model.predict(
            [[current_loan_amount,
              label_encoder_dict["term_label_encoder"].transform([term])[0],
              credit_score, annual_income, years_in_current_job,
              label_encoder_dict["home_label_encoder"].transform(
                  [home_ownership])[0],
              label_encoder_dict["purpose_label_encoder"].transform([purpose])[
                  0],
              monthly_debt,
              years_of_credit_history, number_of_open_accounts,
              number_of_credit_problems, current_credit_balance,
              maximum_open_credit,
              bankruptcies, tax_liens]])
        loan_status = label_encoder_dict[
            "loan_status_label_encoder"].inverse_transform(loan_status)
        if loan_status[0] == "Fully Paid":
            st.write("Customer is eligible for the Loan")
        else:
            st.write(f"Customer is likely to be Charged off")
else:
    col[0].subheader("Upload a File")
    uploadedFile = col[0].file_uploader("", type=['csv', 'xlsx'],
                                        accept_multiple_files=False,
                                        key="fileUploader")
    columns = ["Customer ID", "Current Loan Amount", "Term", "Credit Score",
               "Annual Income", "Years in current job", "Home Ownership",
               "Purpose", "Monthly Debt", "Years of Credit History",
               "Number of Open Accounts", "Number of Credit Problems",
               "Current Credit Balance", "Maximum Open Credit", "Bankruptcies",
               "Tax Liens"
               ]
    if uploadedFile is not None:
        df = pd.read_csv(uploadedFile)
        df.dropna(inplace=True)

        try:
            input_df = df[columns]
            input_df["Years in current job"] = input_df[
                "Years in current job"].apply(
                lambda x: int(x.lower().strip("< + years")) if isinstance(x, (
                    str)) else x)
            input_df["Term"] = input_df["Term"].apply(
                lambda x: label_encoder_dict["term_label_encoder"].transform([x])[0]
            )

            input_df["Home Ownership"] = input_df["Home Ownership"].apply(
                lambda x: label_encoder_dict["home_label_encoder"].transform([x])[0]
            )
            input_df["Purpose"] = input_df["Purpose"].apply(
                lambda x: label_encoder_dict["purpose_label_encoder"].transform([x])[0]
            )
            input_df["Loan Status Prediction"] = xgb_model.predict(input_df.drop("Customer ID", axis=1))
            input_df["Loan Status Prediction"] = input_df["Loan Status Prediction"].apply(
                lambda x: label_encoder_dict["loan_status_label_encoder"].inverse_transform([x])[0]
            )
            st.write(input_df[["Customer ID", "Loan Status Prediction"]])

            show_analysis = st.checkbox("Show Detailed Analysis")
            if show_analysis:
                home_ownership_count = df["Home Ownership"].value_counts()
                home_ownership_count = pd.DataFrame(
                    {'Home Ownership': home_ownership_count.index.tolist(),
                     'Count': home_ownership_count.values}
                )
                fig_home_ownership = px.bar(home_ownership_count,
                                            x="Home Ownership", y="Count")

                purpose_count = df["Purpose"].value_counts()
                purpose_count = pd.DataFrame(
                    {'Purpose': purpose_count.index.tolist(),
                     'Count': purpose_count.values}
                )
                fig_purpose = px.bar(purpose_count, x="Purpose",
                             y="Count")

                status_count = input_df["Loan Status Prediction"].value_counts()
                status_count = pd.DataFrame(
                    {'Loan Status': status_count.index.tolist(),
                     'Count': status_count.values}
                )
                fig_status = px.bar(status_count, x="Loan Status",
                                    y="Count")

                st.plotly_chart(fig_purpose)
                st.plotly_chart(fig_home_ownership)
                st.plotly_chart(fig_status)

                st.dataframe(df)

        except KeyError:
            st.error("Requirement Fields not Present")
            st.stop()
        except Exception as e:
            print(e)
            st.error("Ooops! Something Went Wrong")
            st.stop()




#
# LR = joblib.load('./LR.pkl')
# PR = joblib.load('./PR.pkl')
# XGB = joblib.load('./XGB.pkl')
#
# pr = st.button("Predict")
# if pr:
#     if option == 'Linear Regression':
#         st.write(LR.predict([[amb_temp, solr_radiation, wind_speed]]))
#     elif option == 'Polynomial Regression(Degree 2)':
#         st.write(PR.predict([[amb_temp, solr_radiation, wind_speed]]))
#     elif option == 'XGBModel':
#         st.write(XGB.predict(np.array([[amb_temp,  solr_radiation, wind_speed]])))
