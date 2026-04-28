
**Credit Card Fraud**  

**1.	Problem Statement**
   
Digital payments are evolving, but so are cyber criminals.

According to the Data Breach Index, more than 5 million records are being stolen on a daily basis, a concerning statistic that shows - fraud is still very common both for Card-Present and Card-not Present type of payments.

In today’s digital world where trillions of Card transaction happens per day, detection of fraud is challenging.

These notebooks attempt to provide machine learning and artificial intelligence models to detect fraudulent credit card transactions.

**2.	Model Outcomes or Predictions**
   
The type of learning is classification. The expected output of the selected model is the prediction of credit card fraud. 

Supervised machine learning algorithms are used to build predictive models. 

**3.	Data**
   
A sample of 30,000 rows is obtained from the Credit Card Transactions Dataset that comes from Kaggle.

The dataset provides detailed records of credit card transactions, including information about transaction times, amounts, and associated personal and merchant details.

The dataset can be used in the following ways:

Fraud Detection : Use machine learning models to identify fraudulent transactions by examining patterns in transaction amounts, locations, and user profiles. Enhancing fraud detection systems becomes feasible by analyzing behavioral patterns.

Customer Segmentation : Segment customers based on spending patterns, location, and demographics. Tailor marketing strategies and personalized offers to these different customer segments for better engagement.

Transaction Classification : Classify transactions into categories such as grocery or entertainment to understand spending behaviors. This helps in improving recommendation systems by identifying transaction categories and preferences.

Geospatial Analysis : Analyze transaction data geographically to map spending patterns and detect regional trends or anomalies based on latitude and longitude.

Predictive Modeling : Build models to forecast future spending behavior using historical transaction data. Predict potential fraudulent activities and financial trends.

Behavioral Analysis : Examine how factors like transaction amount, merchant type, and time influence spending behavior. Study the relationships between user demographics and transaction patterns.

Anomaly Detection : Identify unusual transaction patterns that deviate from normal behavior to detect potential fraud early. Employ anomaly detection techniques to spot outliers and suspicious activities.

**4.	Data Preprocessing/Preparation**
   
Leading and trailing white spaces are removed from categorical columns. Any duplicates are verified.    

The missing values in the column 'merch_zipcode' are not imputed to preserve any fraud signals from these missing values.     

Data is cleaned and data types assigned as required.

Feature engineering included the following:

1. User zip codes are regionalized to reduce cardinality

2. Time since last purchase is calculated using 'unix_time'

3. The amount of deviation from the average transactions is found

4. The distance of the user from the merchant is calculated

5. The transaction velocity in miles per hour is computed

6. Age is determined from the 'dob'

7. Hour is expressed in terms of the sine and cosine functions

8. The frequency of merchant transactions is determined

9. 'missing_merch_zip' feature is created from the 'merch_zipcode' missing values

Fraud signals are analyzed using box plots, histograms, stacked bar plots, heat maps and pair plots as well as other visualization techniques like scatter plots and bar charts. 

**5.	Modeling**
   
Several models are built to predict fraudulent transactions. These are in separate notebooks.

Four (4) supervised machine learning classification algorithms are used to build predictive models:  
`DecisionTreeClassifier, Keras Classifier, RandomForestClassifier` and `XGB Classifier`. 

The parameter 'class_weight="balanced"' is used to address the class imbalance as well as the use of 'scale_pos_weight'.

Categorical features are encoded.
    
The features are scaled as required.      

The model parameters are optimized using the appropriate grid search function. 

**Profit/Loss analysis**

The profit/loss analysis compares the relative performance of the models financially. For demonstration purposes, the following values are used:

L = value of fraud is $800     
C = cost of preventive action is $40     
True positives (TP) are correctly identified frauds (preventive action is spent but the value is saved).     
False positives (FP) are not fraudulent but predicted to be fraudulent (preventive action is spent but no value is saved).     
False negatives (FN) are predicted not to be fraudulent but actually fraudulent (no prevention action spent but value is lost).     
True negatives (TN) are correctly predicted not to be fraudulent (no prevention action spent and no value is lost).

The threshold that will maximize profit is determined and used.

**Feature Selection**     

Feature selection reduces noise and improves the interpretation of feature importance.
Feature selection is about removing noise, improving interpretability, improving logistic regression stability, and slightly reducing overfitting.

The model performance metrics and feature importances are output and compared with other models in a separate notebook.

Sample predictions using the model are demonstrated.  

**False Negatives** analysis is made to improve future feature engineering and reduce these False Negatives.

**6.	Model Evaluation**
   
The GenAI model provided the following feature importance analysis on the results of the top three (3) classification models which are averaged, namely, **XGBoost Classifier**, **Decision Tree Classifier**, and **Random Forest Classifier**.

The GenAI model also recommended ways for future feature engineering to improve the model.

"Feature Importance Analysis: Credit Card Fraud Detection ### Overview The feature importance data provided contains the average importance values of various features used in a machine learning model to detect credit card fraud. The features are ranked based on their contribution to the model's performance, with higher values indicating greater importance. ### Key Observations 1. amt_log is the most important feature, contributing about 1.0 to the model's performance. This suggests that the log-transformed transaction amount is a key predictor of fraud. 2. high_amt is the second most important feature, with an average importance of 0.658. This feature might indicate high-value transactions that are more likely to be fraudulent. 3. The top three features (amt_log, high_amt, and amt_vs_avg) are all related to transaction amounts, suggesting that the model is paying close attention to these factors when predicting fraud. 4. The next group of important features include hour_cos (0.267), hour_sin (0.143), and city_pop (0.169). These features might be used to capture temporal and spatial patterns in transaction behavior that are indicative of fraud. ### Insights for Model Building 1. Focus on transaction amount features: The high importance of amt_log and high_amt suggests that transaction amounts are a strong predictor of fraud. Consider incorporating more features related to transaction amounts, such as transaction value distributions or averages. 2. Consider temporal patterns: The presence of hour_cos and hour_sin in the top features indicates that temporal patterns in transactions may be useful for detecting fraud. You could explore additional features that capture temporal dynamics, such as day of the week or time of day. 3. Look for spatial patterns: The importance of city_pop suggests that geographic location may be a factor in predicting fraud. Consider incorporating features that capture regional differences in transaction behavior. ### Code to Extract Insights You can use the following Python code to extract insights from this feature importance data: python import pandas as pd # Define the feature importance data feature_importance = { 'feature': ['amt_log', 'high_amt', 'amt_vs_avg', 'hour_cos', 'city_pop', 'hour_sin', 'gender_M', 'gender_F', 'age', 'travel_speed'], 'average_importance': [1.0, 0.658, 0.505, 0.267, 0.169, 0.143, 0.128, 0.112, 0.105, 0.088] } # Create a DataFrame from the feature importance data df = pd.DataFrame(feature_importance) # Sort the DataFrame by average importance in descending order df = df.sort_values(by='average_importance', ascending=False) # Print the top 5 features with the highest importance print(df.head(5)) This code will output the top 5 features with the highest importance, which can be used to guide model building and feature engineering efforts."

**Overall Model Summary**     

Based on the metrics of  AUC, Accuracy, Precision, Recall, F2 Score, and Profit/Loss, the **XGBoost Classifier** is the winner.  **Decision Tree Classifier** and **Random Forest Classifier** are virtually tied for the second place.

For this particular dataset, the **XGBoost Classifier**  is the recommended machine learning algorithm.    

The **feature importance** is consistent with what is highlighted in the **Heatmap of Significant Correlations**: `amt_log`, `high_amt`, and  `amt_vs_avg`

| Model | AUC | Accuracy | Precision | Recall | F2 Score |Profit/Loss |
|:---------|:---------|:---------|:---------|:---------|:---------|:---------|
| Decision Tree | 0.929 | 0.975 | 0.238 | 0.865 | 0.567 |  22840 | 
| Random Forest  | 0.979 | 0.954 | 0.150 | 0.923 | 0.455 | 22400 | 
| Keras | 0.939 | 0.966 | 0.168 | 0.731 | 0.438 | 10160 | 
| XGBoost  | 0.997 | 0.989 | 0.445 | 0.924 | 0.770 | 32400 | 


**Next Steps and Further Recommendations**  

- Confirm the model that will suit the business needs in terms of the optimal level of fraud identification and precision.
    
- Continue model development to include actual identification of fraudulent transactions and reduce **False Negatives** through improved feature engineering.

- Tune the thresholds to maximize profit using realistic lifetime value and cost of retention assumptions.

- Deploy and apply the model for the use of relevant business groups tasked with prevention of credit card fraudulent transactions. Leadership can also be guided by the model for fraud prevention strategy.     

- Continue model development to validate the features relative importance to guide management on which features need to be given particular attention in order to prevent fraud.

**Notebook**    
You can view the full analysis here:

[Exploratory Data Analysis]
https://github.com/RonaldoBantayan27/Liam/blob/main/01_EDA_Credit_Card_Fraud.ipynb
[XGB Classifier]
https://github.com/RonaldoBantayan27/Liam/blob/main/02_XGB_Credit_Card_Fraud.ipynb
[RandomForestClassifier]
https://github.com/RonaldoBantayan27/Liam/blob/main/03_RF_Credit_Card_Fraud.ipynb
[DecisionTreeClassifier]
https://github.com/RonaldoBantayan27/Liam/blob/main/04_DT_Credit_Card_Fraud.ipynb
[Keras Classifier]
https://github.com/RonaldoBantayan27/Liam/blob/main/05_Keras_Credit_Card_Fraud.ipynb
[Summary]
https://github.com/RonaldoBantayan27/Liam/blob/main/06_Summary_Credit_Card_Fraud.ipynb

**Reference:** 
Ronaldo Bantayan (Author) Email: one01bant@yahoo.com     


