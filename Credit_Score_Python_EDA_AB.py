#!/usr/bin/env python
# coding: utf-8

# In[250]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statistics


# In[251]:


df = pd.read_csv('D:\\Scaler\\Scaler\\Fintech Domain Course\\Credit Python EDA\\Credit_score.csv')


# In[252]:


df.head()


# In[253]:


df.shape


# In[254]:


df.columns


# In[255]:


df.info()


# In[256]:


df.isna().sum()


# In[257]:


df.isna().mean()*100


# In[258]:


df.describe(exclude=np.number).T


# In[259]:


df['Age'].value_counts()


# ### Observations:
# 1. Customer_ID has 12500 unique values. It means we have data of 12500 customers.
# 2. Month has only 8 unique values. Better to analyse further which months are present.
# 3. Age has 1788 unique values. This looks strange as general age range is from 0-100.
# 4. SSN has 12501 unique values, whereas Customer_ID only has only 12500 unique values. There is a possibility that incorrect SSN 5. value is entered for one of the customer as same person can't have multiple SSN.

# # Step 1: Buidling Common Functions for Data Cleaning

# #### Analysing Data: 1: Getting details of columns (Features) including data type, null values, unique values and value counts

# In[260]:


def column_info(df,column):
    print("Details of",column,"column")
    
    #DataType of a column
    print("\nDataType: ",df[column].dtype)
    
    #Checking for null values
    count_null = df[column].isnull().sum()
    if count_null==0:
        print("\nThere are no null values")
    elif count_null>0:
        print("\nThere are ", count_null," null values")
        
    #Checking for Unique Values
    print("\nNumber of Unique Values: ",df[column].nunique())
    
    #Checking for value counts    
    print("\n Series of Unique Values:\n")
    print(df[column].value_counts())


# #### Feature Engineering for Numerical columns: 1. Filling Missing values with 'mode'

# In[261]:


def feat_eng1_num_replace_with_mode(df, groupby, column):
    print("\n No. of missing values before feature engineering: ", df[column].isnull().sum())
    
    #Filling process with mode 
    mode_process = df.groupby(groupby)[column].transform(lambda x: x.mode().iat[0])
    #mode_process = df.groupby(groupby)[column].transform(lambda x: x.mode(keepdims=True).iat[0])

    
    df[column] = df[column].fillna(mode_process)
    
    print("\n No. of missing values after feature engineering: ", df[column].isnull().sum())


# #### Feature Engineering for Numerical columns: 2. Handling Outliers and null values together

# In[262]:


def feat_eng2_replace_outliers_null(df, groupby, column):  
    print("\n Min, Max Values:", df[column].apply([min, max]), sep='\n', end='\n')   
    
    df_dropped = df[df[column].notna()].groupby(groupby)[column].apply(list)
    x, y = df_dropped.apply(lambda x: stats.mode(x)).apply([min, max])
    mini, maxi = x[0][0], y[0][0]
    
    # Replace with NaN to outliers
    col = df[column].apply(lambda x: np.NaN if ((x<mini)|(x>maxi)|(x<0)) else x)
    
    # fill with mode values
    mode_by_group = df.groupby(groupby)[column].transform(lambda x: x.mode()[0] if not x.mode().empty else np.NaN)

    #mode_by_group = df.groupby(groupby)[column].transform(lambda x: x.mode(keepdims=True)[0] if not x.mode().empty else np.NaN)

    df[column] = col.fillna(mode_by_group)
    
    #Filling Remaining NaN Values with Mean
    df[column].fillna(df[column].mean(), inplace=True)    
    
    print("\n After data Cleaning Min, Max Values:", df[column].apply([min, max]), sep='\n', end='\n') 
    print("\n No. of Unique values after Cleaning:",df[column].nunique())
    print("\n No. of Null values after Cleaning:",df[column].isnull().sum())


# #### Feature Engineering for Numerical columns: 3. Removing undefined/garbage values

# In[263]:


def feat_eng3_num_replace_undefinedVal(df, groupby, column, strip=None, datatype=None, replace_value=None):
    #Replace with np.nan
    if replace_value != None:
        df[column] = df[column].replace(replace_value, np.nan)
        print(f"\n Undefined value {replace_value} is replaced with np.nan")
    
    # Remove trailing & leading special characters
    if df[column].dtype == object and strip is not None:
        df[column] = df[column].str.strip(strip)
        print(f"\nTrailing & leading {strip} are removed")
        
    # Change datatype
    if datatype is not None:
        df[column] = df[column].astype(datatype)
        print(f"\nDatatype of {column} is changed to {datatype}")    
        
        
    feat_eng2_replace_outliers_null(df, groupby, column)   


# #### Feature Engineering for Categorical columns: 1. Replacing with null values or filling Missing values with 'mode'

# In[264]:


def feat_eng4_cat_replace_with_null_mode(df, groupby, column, replace_value = None):
    print("\n Cleaning Categorical column: ", column)
    
    #Replace with null values
    if replace_value != None:
        df[column] = df[column].replace(replace_value, np.nan)
        
    feat_eng1_num_replace_with_mode(df, groupby, column)    


# # Buidling Common Functions for Data Visualization

# In[265]:


#countplot

def plot_countplot(df, column, edited_column, rotation=0):
    print(f'\n{edited_column} Distribution')
    
    palette = "deep" 
    sns.set_palette(palette)
    
    sns.countplot(data=df, x=column)

    plt.xlabel(f'{edited_column}')
    plt.ylabel('Number of Records')
    plt.title(f'{edited_column} Distribution')
    plt.xticks(rotation=rotation)

    plt.show()


# In[266]:


#displot

def plot_displot(df, column, edited_column, rotation=0, bins=20):
    print(f'\n{edited_column} Distribution')
    palette = "deep" 
    sns.set_palette(palette)
    
    sns.displot(data=df, x=column, kde=True, bins=bins)

    plt.xlabel(f'{edited_column}')
    plt.ylabel('Number of Records')
    plt.title(f'{edited_column} Distribution')
    plt.xticks(rotation=rotation)

    plt.show()


# In[267]:


#stackedbar

def plot_stacked_bar(df, column1, column2, rotation=0):
    print(f'\n{column1} & {column2} Distribution')
    palette = "deep" 
    sns.set_palette(palette)

    pd.crosstab(df[column1], df[column2]).plot(kind='bar', stacked=True)
    
    plt.xlabel(f'{column1}')
    plt.ylabel('Number of Records')
    plt.title(f'{column1} & {column2} Distribution')
    plt.xticks(rotation=rotation)

    plt.show()


# ## Analysing Important Features for data cleaning

# In[268]:


#getting details of ID

column_info(df,'ID')


# In[269]:


#getting details of ID

column_info(df,'Customer_ID')


# In[270]:


#getting details of Month

column_info(df,'Month')


# In[271]:


#Convert Month to datetime object
df['Month'] = pd.to_datetime(df.Month, format='%B').dt.month


# In[272]:


df.columns


# In[273]:


groupby = 'Customer_ID'
column = 'Name'


feat_eng4_cat_replace_with_null_mode(df, groupby, column, replace_value = None)


# In[274]:


#getting details of SSN

groupby = 'Customer_ID'
replace_value = '#F%$D@*&8'
column ='SSN'

column_info(df,'SSN')

feat_eng4_cat_replace_with_null_mode(df, groupby, column, replace_value)


# In[275]:


#Get Details of Type of Loan column
column_info(df,'Type_of_Loan')


# In[276]:


df['Type_of_Loan'].replace([np.NaN], 'Not Specified', inplace=True)


# In[277]:


df['Type_of_Loan'].value_counts()


# In[278]:


column_name = 'Credit_Mix'

#Get Details of Type of Credit_Mix column

column_info(df,'Credit_Mix')


# In[279]:


# Data Cleaning
column = 'Credit_Mix'
groupby = 'Customer_ID'
replace_value = '_'

feat_eng4_cat_replace_with_null_mode(df, groupby, column, replace_value)


# In[280]:


#Get Details
column_info(df,'Payment_Behaviour')


# In[281]:


column = 'Payment_Behaviour'
groupby = 'Customer_ID'
replace_value = '!@9#%8'


# In[282]:


feat_eng4_cat_replace_with_null_mode(df, groupby, column, replace_value)


# ##  Numerical Features

# In[283]:


column = 'Age'

edited_column = 'Age'
#Get Details
column_info(df,'Age')


# In[284]:


#Plot Graph
#plot_displot(df,'Age','Age', bins=40)
plot_displot(df, column, edited_column, rotation=0, bins=20)


# In[285]:


groupby = 'Customer_ID'
column = 'Age'

#Cleaning

feat_eng3_num_replace_undefinedVal(df, groupby, column, strip='_', datatype=int)


# In[286]:


column = 'Annual_Income'
groupby = 'Customer_ID'
edited_column = 'Annual Income'

#Get Details
column_info(df,'Annual_Income')


# In[287]:


#Cleaning
feat_eng3_num_replace_undefinedVal(df, groupby, column, strip='_', datatype=float)


# In[288]:


plot_displot(df,column,edited_column,bins=40)


# In[289]:


column = 'Num_Bank_Accounts'
edited_column = 'Number of Bank Accounts'
group_by = 'Customer_ID'


# In[290]:


#Get Details
column_info(df,'Num_Bank_Accounts')


# In[291]:


#Cleaning
feat_eng3_num_replace_undefinedVal(df, groupby, column)


# In[292]:


plot_displot(df,column,edited_column,bins=40)


# # Feature Engineering

# ## In order to calculate Credit Score, We will use the following compenents and weightage:
#     ** (Payment_history * 0.35) + 
#     ** (credit utilization ratio * 0.15) + 
#     ** (Monthly_Debt_to_Income_Ratio * 0.15) +
#     ** (No. of credit card a/c * 0.15) + 
#     ** (Employment Status (Occupation) * 0.10) + 
#     ** (Credit History Age * 0.10)   
# 

# ###### Payment History (Payment_history * 0.35):
# Reasoning: Payment history is a fundamental factor in assessing creditworthiness. Timely payments contribute positively to the credit score, while delayed or missed payments can have a negative impact. Assigning the highest weight (0.35) to payment history reflects its significance in predicting an individual's ability to manage debt responsibly.
# 
# ###### Credit Utilization Ratio (credit utilization ratio * 0.15):
# Reasoning: The credit utilization ratio is the ratio of current credit card balances to credit limits. A low credit utilization ratio is generally considered favorable, indicating responsible credit usage. By assigning a weight of 0.30, we acknowledge the importance of maintaining a healthy balance between available credit and credit usage in determining creditworthiness.
# 
# ###### Number of Credit Card Accounts (No. of credit card a/c * 0.15):
# Reasoning: The number of active credit card accounts provides insights into an individual's credit management. Having a reasonable number of credit card accounts can positively impact credit scores. Assigning a weight of 0.15 acknowledges the role of credit diversity and responsible credit card usage in the overall creditworthiness assessment.
# 
# ###### Monthly Debt to Income Ratio (No. of credit card a/c * 0.15):
# Reasoning: Monthly Debt to Income Ratio provides insights into an individual's financial health. A lower ratio indicates that a person has more disposable income after meeting their debt obligations, which is generally considered a positive factor.
# 
# ###### Employment Status (Occupation) (Employment Status * 0.10):
# Reasoning: Employment status, represented by the Occupation column, is considered in assessing stability and financial capability. Certain occupations may indicate a steady income and job security. Assigning a weight of 0.10 recognizes the influence of employment status on an individual's ability to meet financial obligations.
# 
# ###### Credit History Age (Credit History Age * 0.10):
# Reasoning: The age of credit history reflects the length of time an individual has been using credit. A longer credit history is generally viewed positively as it provides a more extended track record of credit management. Assigning a weight of 0.10 recognizes the importance of a well-established credit history in determining creditworthiness.
# 
# In summary, the chosen components and their respective weightages aim to capture key aspects of an individual's financial behavior, responsible credit usage, and stability. The weights are assigned based on the relative impact of each component on predicting creditworthiness, as well as industry standards and best practices in credit scoring.

# ## 1. Data Preparation for Payment_History
# We will use 
# 
#     a. Delay_from_due_date
#     b. Num_of_Delayed_Payment
#     c. Payment_of_Min_Amount
# 
# to create new feature - Payment_Histroy

# In the context of credit score calculation, the Payment_History feature is a crucial component that reflects an individual's creditworthiness based on their past payment behavior. The choice of using Delay_from_due_date, Num_of_Delayed_Payment and Payment_of_Min_Amount for creating the Payment_History feature is rooted in the following reasoning:
# 
# Delay_from_due_date: 
# Timeliness of Payments: Delay_from_due_date provides information about the delay in making payments from the due date. Timely payments are a crucial indicator of financial responsibility and discipline.
# Impact on Creditworthiness: A consistent history of delayed payments negatively affects creditworthiness. By including this component, the credit score model captures the historical pattern of payment delays.
# 
# Num_of_Delayed_Payment: 
# Frequency of Delays: The number of delayed payments (Num_of_Delayed_Payment) reflects the frequency of instances where a borrower failed to make payments on time.
# Risk Assessment: A higher number of delayed payments indicates a higher risk of default and financial instability. Including this component helps lenders assess the level of risk associated with a borrower's payment behavior.
# 
# Payment_of_Min_Amount: 
# Meeting Minimum Obligations: Payment_of_Min_Amount signifies whether the borrower consistently meets at least the minimum payment obligations. This is essential for maintaining a positive credit history.
# Credit Management Discipline: Borrowers who consistently pay at least the minimum amount due demonstrate a certain level of credit management discipline. Including this component contributes to a more comprehensive evaluation of payment behavior.
# 

# ### a: Delay_from_due_date

# In[293]:


column = 'Delay_from_due_date'
edited_column = 'Delay from Due Date'
groupby = 'Customer_ID'

column_info(df,column)


# In[294]:


feat_eng3_num_replace_undefinedVal(df, groupby, column)


# In[295]:


plot_displot(df, column, edited_column, rotation=0, bins=20)


# ### b: Num_of_Delayed_Payment

# In[296]:


column = 'Num_of_Delayed_Payment'
edited_column = 'Number of Delayed Payment'

column_info(df,column)


# In[297]:


groupby = 'Customer_ID'
feat_eng3_num_replace_undefinedVal(df, groupby, column, strip='_', datatype='float')


# In[298]:


plot_countplot(df, column, edited_column, rotation=90)


# ### c: Payment_of_Min_Amount

# In[299]:


column = 'Payment_of_Min_Amount'

#Get Details
column_info(df,column)


# In[300]:


#Implementing Label encoding Feature
df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace({"Yes": 1, "No": 0, "NM": 0})


# In[301]:


df["Payment_History_Score"] = (
      -1 * df["Delay_from_due_date"]
      -1 * df["Num_of_Delayed_Payment"]
      + 1 * df["Payment_of_Min_Amount"]
  )


# In[302]:


df[["Payment_History_Score"]]


# ## 2. Data Preparation for Credit_Utilization_Ratio

# We will use Credit_Utilization_Ratio column directly from the data-set

# In[303]:


column = 'Credit_Utilization_Ratio'
edited_column = 'Credit Utilization Ratio'

column_info(df,column)


# In[304]:


plot_displot(df, column, edited_column)


# ## 3. Data Preparation for Monthly_Debt_to_Income_Ratio

# We will use
# 
#     a. Outstanding_Debt
#     b. Monthly_Inhand_Salary
#     
# to create new feature - Monthly_Debt_to_Income_Ratio

# ### a: Outstanding_Debt

# In[305]:


column = 'Outstanding_Debt'
groupby = 'Customer_ID'
edited_column = 'Outstanding Debt'

#Get Details
column_info(df,column)

#Cleaning
feat_eng3_num_replace_undefinedVal(df, groupby, column, strip='_',datatype=float)


#Plot Graph
plot_displot(df,'Outstanding_Debt', 'Outstanding Debt', rotation=90)


# ### b: Monthly_Inhand_Salary

# In[306]:


column = 'Monthly_Inhand_Salary'
groupby = 'Customer_ID'

#Get Details
column_info(df,'Monthly_Inhand_Salary')


# In[307]:


#Cleaning
feat_eng3_num_replace_undefinedVal(df, groupby, column)


# In[308]:


edited_column = 'Monthly_Inhand_Salary'
plot_displot(df, column, edited_column, bins=40)


# In[309]:


# Calculating Debt to Income ratio
df['Monthly_Debt_to_Income_Ratio'] = df['Outstanding_Debt'] / df['Monthly_Inhand_Salary']


# In[310]:


df[['Monthly_Debt_to_Income_Ratio']]


# ## 4. Data Preparation for Num_Credit_Card

# In[311]:


column = 'Num_Credit_Card'
edited_column = 'Number of Credit Card'

column_info(df,column)


# In[312]:


groupby = 'Customer_ID'
feat_eng3_num_replace_undefinedVal(df, groupby, column)


# In[313]:


plot_countplot(df,column,edited_column)


# ## 5. Data Preparation for Employment Status

# ** As In credit scoring models, employment status is often considered as a significant factor because it provides insights into an individual's financial stability and ability to repay debts. Individuals with stable employment are generally considered less risky borrowers. However, the given dataset doesn't have a specific 'Employment_Status' column, so we use the 'Occupation' column as a proxy for employment status
# 
# ** 'Occupation' can serve as a proxy for employment status, as certain occupations are associated with stable employment (e.g., doctors, teachers) while others may indicate more variable or entrepreneurial sources of income.

# In[314]:


column = 'Occupation'
groupby = 'Customer_ID'

#getting details of Occupation

column_info(df,'Occupation')


# In[315]:


replace_value = '_______'
#user_friendly_name = 'Occupation'

feat_eng4_cat_replace_with_null_mode(df, groupby, column, replace_value)


# In[316]:


# Map occupation categories to employment status weights
employment_status_weights = {
    'Lawyer': 0.10,
    'Architect': 0.10,
    'Engineer': 0.10,
    'Scientist': 0.10,
    'Mechanic': 0.05,
    'Accountant': 0.10,
    'Developer': 0.10,
    'Media_Manager': 0.10,
    'Teacher': 0.10,
    'Entrepreneur': 0.10,
    'Doctor': 0.10,
    'Journalist': 0.05,
    'Manager': 0.10,
    'Musician': 0.05,
    'Writer': 0.05
}


# In[317]:


#df['Occupation'].map(employment_status_weights).fillna(0) * weights['Employment_Status']
# Calculate Employment_Status component
df['Employment_Status'] = df['Occupation'].map(employment_status_weights).fillna(0)


# In[318]:


# Display the resulting DataFrame
df[['Occupation', 'Employment_Status']]


# ## 6. Data Preparation for Credit History Age

# In[319]:


df['Credit_History_Age'].value_counts()


# In[320]:


def Month_Converter(val):
    if pd.notnull(val):
        years = int(val.split(' ')[0])
        month = int(val.split(' ')[3])
        return (years*12)+month
    else:
        return val


# In[321]:


df['Credit_History_Age'] = df['Credit_History_Age'].apply(lambda x: Month_Converter(x)).astype(float)


# In[322]:


df[['Credit_History_Age']]


# In[323]:


column = 'Credit_History_Age'

column_info(df,column)


# In[324]:


groupby = 'Customer_ID'
feat_eng3_num_replace_undefinedVal(df, groupby, column, datatype=float)


# In[325]:


edited_column = 'Credit History Age'
#Plot Graph
plot_displot(df, column, edited_column)


# In[326]:


df[['Payment_History_Score', 'Credit_Utilization_Ratio', 'Monthly_Debt_to_Income_Ratio', 'Num_Credit_Card','Employment_Status', 'Credit_History_Age' ]]


# ## Calcuting Credit Score 
# ### 1. Group by Customer ID, handling month-level data and calculating scores
# ### 2. Standardize values for numerical features
# ### 3. Calculate weighted scores
# ### 4. Normalize scores to a range of 0 to 100

# In[327]:


def calculate_credit_score(data):
    # Group by Customer ID, handling month-level data and calculating scores
    grouped_data = data.groupby("Customer_ID").agg(
    Payment_History_Score=("Payment_History_Score", "mean"),
    Credit_Utilization_Ratio=("Credit_Utilization_Ratio", "mean"),  
    Monthly_Debt_to_Income_Ratio=("Monthly_Debt_to_Income_Ratio", "mean"),
    Num_Credit_Card=("Num_Credit_Card", "mean"),
    Employment_Status=("Employment_Status", "mean"),
    Credit_History_Age=("Credit_History_Age", "max"),  # Using maximum age as it seems to be a measure of history duration
    )


    # Standardize values for numerical features
    grouped_data = (grouped_data - grouped_data.mean()) / grouped_data.std()

    # Calculate weighted scores
    grouped_data["credit_score"] = (
      0.35 * grouped_data["Payment_History_Score"]
      + 0.15 * (1-grouped_data["Monthly_Debt_to_Income_Ratio"]) #Inverse relation as lower the value better the financials
      + 0.15 * (1-grouped_data["Credit_Utilization_Ratio"]) #inverse relation
      + 0.15 * (grouped_data["Num_Credit_Card"]) 
      + 0.10 * grouped_data["Employment_Status"]
      + 0.10 * grouped_data["Credit_History_Age"]                      
     )

     # Normalize scores to a range of 0 to 100
    grouped_data["credit_score"] = (grouped_data["credit_score"] - grouped_data["credit_score"].min()) / (grouped_data["credit_score"].max() - grouped_data["credit_score"].min()) * 100

    return grouped_data.reset_index()


# In[328]:


# Calculate scores for all customers
credit_scores_df = calculate_credit_score(df)
credit_scores_df[["Customer_ID","credit_score"]]


# In[329]:


# Assuming 'result_df' is your DataFrame with the calculated credit scores
# Adjust the percentile values as needed
good_threshold = credit_scores_df['credit_score'].quantile(0.75)
poor_threshold = credit_scores_df['credit_score'].quantile(0.25)

# Create a new column 'Credit_Score_Category'
credit_scores_df['Credit_Score_Category'] = pd.cut(
    credit_scores_df['credit_score'],
    bins=[-float('inf'), poor_threshold, good_threshold, float('inf')],
    labels=['Poor', 'Standard', 'Good'],
    include_lowest=True
)




# In[330]:


credit_scores_df


# In[331]:


column = 'Credit_Score_Category'
edited_column = 'Credit Score'

#Get Details
column_info(credit_scores_df,column)


# In[332]:


#Plot Graph
plot_countplot(credit_scores_df,column,edited_column)


# ## Summary
# 
# Insights from the dataset reveal that individual customer data is available for an 8-month period spanning from January to August. The dataset includes various loan types, such as auto loans, credit-builder loans, debt consolidation loans, home equity loans, mortgage loans, payday loans, personal loans, and student loans. 
# 
# A notable trend is observed in the customers' annual income, which predominantly exhibits a right-skewed distribution, indicating that most customers have lower annual incomes.
# 
# Furthermore, the analysis of monthly income distribution follows a similar pattern, with a predominant right-skewed trend among customers. Regarding the number of bank accounts, the majority of customers maintain between 3 to 8 accounts. The distribution of the number of credit cards spans from 0 to 11, with a concentration between 3 to 7, peaking at 5.
# 
# Interest rates on loans vary across the dataset, ranging from 1% to 34%. The delay from the due date is observed to be concentrated within the 0 to 30 days range. Notably, only a limited number of customers invest amounts exceeding 2,000 per month. When it comes to the number of loans taken by customers, the typical range falls between 2 to 4 loans, with a maximum recorded at 9. These insights offer a comprehensive overview of the financial dynamics and behaviors observed in the dataset.
# 

# ## Advantage/Impact of Credit Score: 
# The credit score serves as a vital tool in further financial analysis, providing a concise measure of an individual's creditworthiness. Lenders commonly use credit scores to assess the risk associated with extending credit, determining interest rates, and making lending decisions. A higher credit score often leads to more favorable loan terms, lower interest rates, and increased access to financial products. Additionally, a good credit score can positively influence various aspects of financial life, such as securing housing, obtaining favorable insurance rates, and even impacting employability in certain industries. Therefore, a well-calculated credit score contributes significantly to informed decision-making in financial and lending domains.

# In[ ]:




