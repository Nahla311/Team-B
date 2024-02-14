# Team-B
# Predictive-Modelling-Using-Social-Profile-on-Online-P2P-Lending-Market
We study the borrower-, loan- and group- related determinants of performance predictability in an online P2P lending market by conceptualizing financial and social strength to predict borrower rate and whether the loan would be timely paid.

## SECTION 1: DATA & PROBLEM STATEMENT

Problem statement:
- There is a P2P (peer to peer) lending market, and this company provides loan on basis of their financial strengths and social strengths.
- Financial Factors: - Credit Score - Income Level - Employment Status - Debt-to-Income Ratio - Loan Amount Requested - Previous Loan History - Monthly Expenses.
- Here we need a predict whether the borrowers could be funded with lower interest, and the lenders are receiving the timely payments.
- we have two target columns one is Loan status (categorical) and second one is borrower rate (numerical and continuous).
- In total we need to propose a model by which it should predict the LOAN STATUS.

Data:
- The given dataset contains 113937 records with 81 features which will be used to predict whether a loan and how much of loan amount can be provided to the user.
- The given dataset did not have a target variable and with the finance domain knowledge the target variables where derived. 
- Data duplicate where checked and it proved to be in 0.
- Null values and outlier presence where checked using descriptive statistics.
<img width="608" alt="image" src="https://github.com/shaahidh/p2p-prediction/assets/56645593/52079d56-c6cf-4d72-bd3c-68a227832c2e">

## SECTION 2: EDA

The EDA was conducted before treating outliers and null values and after treating them, to check for the differences it creates.
### EDA before treating outliers and null values:
Univariate and Bi variate analysis was conducted and a better understanding of the data was found some key findings include:
- We observed that we have more Current loans with 56576, such that loans are more in open state than the completed loans are 38074
There are less cancelled loans with 5 and past due loans are with 806+265+313+363+304+16 = 2067 and that will be positive sign.
- people with final payments to done are 205.
- while the Chargedoff (lender's loss - a loan as unlikely to be collected is) with 11992.
- while defaulted is (when the borrower misses payments or fails to fulfill other obligations specified in the loan agreement) and with 5018.
- More people are taking loans for 36 months term, compatively less people are taking loans for 60 months term, very less people are taking loans for 12 months term.
- C (fair or Average)Creditgrade is with high people 5649.
- E (Below subprime)Creditgrade is with 3289.
- The highest number of individuals fall within the $25,000-$49,999 income range, followed closely by those in the $50,000-$74,999 bracket.
- There are a noticeable number of individuals who have chosen not to display their income or are not employed.
  <img width="767" alt="image" src="https://github.com/shaahidh/p2p-prediction/assets/56645593/a4391839-3801-446d-9a10-0af5e28fe6e6">
- The borrower-ARP and borrower-Rate estimated effective yield and estimated return proves to have a positive linear relationship. While the lender yield and estimated loss proves to have a mixed linear relationship.
- Investors mostly invest in the range of the people with debt to income ratio of 0-25%.
![image](https://github.com/shaahidh/p2p-prediction/assets/56645593/964d2abe-f0b0-4f55-bacd-4bd7a3ca3288)
-In this Heat map it explains that BorrowerAPR, BorrowerRate and the LenderYield having high correlation of 0.99 and in the feature we can take only one column instead of all.
Estimated Return w.r.t BorrowerAPR, BorrowerRate and the LenderYield are also giving positive correlation with 0.79,0.82,0.82
ProsperScore, ProsperRating(numeric) having a positive correlation of 0.71

### EDA after outliers and null value treatments:

-We observed that we have more Current loans with 56576, such that loans are more in open state than the completed loans are 38074. 
There are less cancelled loans with 5 and past due loans are with 806+265+313+363+304+16 = 2067 and that will be positive sign
And people with final payments to done are 205.
- It is observed that More people are taking loans for 36 months term are having loans 113937 loans.
- Even with the removal of null and outliers, there is Linear Pattern between BorrowerRate and BorrowerAPR(Annual Percentage Rate), EstimatedReturn and EstimatedEffectiveYield.
![image](https://github.com/shaahidh/p2p-prediction/assets/56645593/074f7f77-6ca3-4484-9e53-621a41fa7171)
- In this Heat map it explains that BorrowerAPR, BorrowerRate and the LenderYield having high correlation of 0.99 and in the feature we can take only one column instead of all.
- Estimated Return w.r.t BorrowerAPR, BorrowerRate and the LenderYield are also giving positive correlation with 0.71,0.0.73,0.74.
- ProsperScore, ProsperRating(numeric) having a positive correlation of 0.71.

Regression & Clustering analysis was also done to understand the relationship between the variables.
![image](https://github.com/shaahidh/p2p-prediction/assets/56645593/ad1310eb-113d-4b9e-b2d7-ea070071d37f)
![image](https://github.com/shaahidh/p2p-prediction/assets/56645593/16d162ea-7ad5-4f86-b4ad-317dfc010fb1)

## SECTION 3: FEATURE SELECTION

Various feature selection techniques where used for feature selection, and each of these methods where used to build a model to find the best one
the feature selection used here:
- RFE
- mutual information
- PCA

## SECTION 4: MODEL BUILDING

### PRE PROCESSING:
Standard scaling, one hot encoding, label encoding were imposed to build a good predicting model.
Target variable for the classification model was created using this function
converting loan status(target variable) to binary category by adding an another column Status:
- def tgt_binary(df):
    - df["Loan_Status"] = df["ClosedDate"].apply(lambda x: 1 if pd.isnull(x) else 0)
    - #df["Loan_Status"] = df["LoanCurrentDaysDelinquent"].apply(lambda x: 1 if x > 180 else 0)
    - return df
- data = tgt_binary(data)
- data.head()

For linear regression three target variables was to be derived using the pre exisitng features.
### Creating EMI colum
-	Tenure ---> **LoanTenure**
-	Principle repayment ---> **LP_CustomerPrinciplePayments**
-	Interest ---> **BorrowerRate** <br />
Formula:
**For each row in the dataset:**
1. Calculate result_1 = P * r * 〖(1+r)〗^n
2. Calculate result_2 = 〖(1+r)〗^n – 1
3. Calculate EMI = result_1 / result_2

### Creating ELA (eligible loan amount):
-	A: “AppliedAmount” ---> **LoanOriginalAmount**
-	R: “Interest” ---> **BorrowerRate**
-	N: “LoanTenure” ---> **LoanTenure**
-	I: “IncomeTotal”  ---> **StatedMonthlyIncome**

**Calculation Procedure:**
**For each row in the dataset:**
1.	Calculate: Total Payment Due = (A + (A*r)) * n
2.	Calculate: Max allowable amount = I * 12 * 30%
3.	If ( Total Payment Due <= Max allowable amount)
- Then ELA = AppliedAmount
- Else ELA = Max allowable amount

### Creating PROI (prefered return of inverstment):
- A: "AppliedAmount" ---> **LoanOriginalAmount** 
- I: "Interest" ---> **BorrowerRate**
- S: "Status" ---> **LoanCurrentDaysDelinquent**
- R: "Rating" ---> **ProsperRating(Alpha)**
- P: "Payment" ---> **MonthlyLoanPayment**
- L: "LoanPayment" ---> **LP_CustomerPrincipalPayments**

**Calculation Procedure:**
**For each row in the dataset:**
1. Calculate: Interest Amount = A * I
2. Calcualte: Total Amount = Interest Amount + A
3. Calculate: ROI = Interest Amount / Total Amount
4. calculate: PROI
**Checking for L**
- If L <= 1000:
     PROI = PROI + 0.05
- elif L > 2000 & L <= 10500
     PROI = PROI - 0.05
- elif L > 10500
     PROI = PROI - 0.1

**Checking for R**
- If R in [2,3] :
     PROI = PROI + 0.05
- elif R = 6:
     PROI = PROI - 0.05

**Checking for A**
- If A <= 2000:
     PROI = PROI - 0.05
- elif A > 19500 & A <= 25500:
     PROI = PROI + 0.05
- elif A > 25500 :
     PROI = PROI + 0.1

**Checking for S**
- If S >= 50 :
     PROI = PROI + 0.05

**Checking for P**
- If P <= 90 :
     PROI = PROI - 0.05
- elif P <= 750 & P > 360 :
     PROI = PROI + 0.05
    
### MODELLING
- Build a classification model to find out whether a loan can be provided or not.
- Build a multi regressor model to find out how much amount we can lend.
these models were evalutated using cross validation method. 
### Building a classification model:
For classification model we chose, random forest,Knn,XG boost, GBM.

All these models where run with the feature selections iteratively and found out XGboost model gave the best performance 
and thus it was selected for the final pipeline.

The model metrics are given below:
- THE BEST MODEL IS XTREME GRADIENT BOOSTING CLASSIFIER with HYPER PARAMETERS OF MAX_DEPTH (max_depth = 7).
- train_accuracy: 0.9999874616011535
- test_accuracy: 0.9943244982739453
- classification Report:               precision    recall  f1-score   support

           0       0.99      1.00      0.99     16579
           1       1.00      0.99      0.99     17603

    accuracy                           0.99     34182
   macro avg       0.99      0.99      0.99     34182
weighted avg       0.99      0.99      0.99     34182

### Building a regressor model:
For regressor model we chose multilinear regressor, multi task lasso regressor (L1 norm).and we found out the best model was linear regressor and thus it was selected for the pipeline.

All these models run with or without the feature selections iteratively and found out multi linear regressor with feature selection gave the best performance and thus it was selected for the final pipeline.

The model metrics are given below:
- THE BEST MODEL IS MULTILINEAR REGRESSOR MODEL WITH THE FEATURE SELECTION GAVE THE BEST PERFORMANCE.
- Mean Squared Error: 848285159.199186
- r2_score: 0.6376675358832489

## SECTION 5: pipeline development:

Three pipeline models has to be created to facilitate smooth operation between all these models <br />
Pipeline 1 was for classification:
- Train test split -----> preprocessing ((numerical & categorical pipeline)standard scaling each of them) -----> classifier (Xgboost) <br />
column transformer where used to combine both the numerical and categorical pipelines <br />
Pipeline 2 was for regression:
- Train test split -----> preprocessing ((numerical & categorical pipeline)standard scaling each of them) -----> regressor (linear regression)<br />
column transformer where used to combine both the numerical and categorical pipelines <br />.
Pipeline 3 was to combine both the results:
- Save both pipelines in a dictionary
- pipeline3 = {'pipeline1': pipeline1, 'pipeline2': pipeline2}

## SECTION 6: Deployment

The P2P lending platform web application is developed using Flask, a Python web framework, enabling users to predict loan statuses and borrower rates through a user-friendly interface.

Flask Application:
- Flask provides the backend functionality, including rendering templates and processing data for predictions.
- The application accepts input through an HTML form and utilizes two machine learning models, a classification model (xgb) and a regression model (lr), serialized with pickle.
- Upon submission of the form, the Flask app collects the data, converts it into a pandas DataFrame, and then passes it to the respective models for prediction.
  
Machine Learning Models
- The classification model predicts the loan status, identifying whether a loan will be in good standing or default.
- The regression model estimates the borrower rate, which could be used to evaluate the interest rate a borrower might be charged.
- Both models were trained on historical data from the platform, considering a range of financial factors and borrower information.
  
Prediction and Results
- Users receive predictions in two categories: loan status (categorical) and borrower rate (numerical and continuous).
The results are displayed on a new HTML page (predictions.html), providing immediate feedback on the potential loan status and the borrower rate.
