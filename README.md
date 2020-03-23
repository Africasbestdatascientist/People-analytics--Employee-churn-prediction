90People-analytics--Employee-churn-prediction
Using a dataset from IBM on employee churn, i created a predictive model to identify employees that are likely to churn. I believe in the era of big data, the use of ML and AI is inevitable in the running of every business. Every business unit needs to adopt ML in its operations, from sales to IT. While the use of ML is popular in some business units, its less applied in others especially in HR. Companies will save millions of dollars if they are able to reduce their employee turnover. Companies with low employee turnover naturally attracts talents into the company compared with companies with high turn over. Hence the need for companies units like HR to adopt the use of ML.  
"/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv", the link to obtaining the data set.
I used 5 different classifiers, Logistic, Naive Baysian, SVM, Random Forrest and XGBoost in modelling. The best classifer with the best AUC was Random forest with an AUC of 95%, accuracy, recall, precision and F1-score were all above 90%. It must be noted that the data set was imbalanced and hence applied SMOTE and ADASYN. ADASYN was better when combined with the classifers. 

Can you do better than that?
