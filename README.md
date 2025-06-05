# Churn-prediction-subscribers

Churn Prediction for Digital News Subscribers:
 Project Overview
 This project predicts churn among digital subscribers of a news platform using synthetic user behavior
 data. It leverages Python for data analysis and machine learning (Logistic Regression, Decision Tree) and
 exports results to Excel for business reporting.
 Objective
 Predict whether a user will unsubscribe (churn) based on: - Engagement metrics - Login activity 
Subscription duration - Support history

 Tools & Technologies
 • Python: pandas, numpy, scikit-learn, matplotlib
 •  ML Models: Logistic Regression, Decision Tree
 •  Excel: Final data export for visualization and reporting

 Python: pandas, numpy, scikit-learn, matplotlib
 ML Models: Logistic Regression, Decision Tree
 Excel: Final data export for visualization and reporting
 📊 Dataset Features
 Simulated dataset with the following fields: - 
 subscription_length : Months subscribed 
 last_login_days_ago : Days since last login 
 articles_read_per_week : Weekly content interaction
 engagement_score: Score between 0–1 (custom metric)  
 support_tickets_raised : Number of support interactions
 churn :  1 if churned, 0 otherwise

 📈 Model Performance
 Logistic Regression
 • Accuracy: ~86.5%
 • Balanced but with a few false positives and false negatives

  Decision Tree (depth=4)
 • Accuracy: 100% (overfit on small data, ideal for understanding split logic)

 How It Works
 1. Simulate 1000 data points based on churn-related heuristics 
 2. Train/test split (80/20)
 3. Fit models
 4. Evaluate predictions with classification report and confusion matrix
 5. Export complete results (with predictions) to Excel



  Output
 • subscriber_churn_predictions.xlsx — contains:
 • All features
 • Ground truth churn 
 • Predicted churn

  
 Business Use Case
 Helps content platforms: - Predict users likely to unsubscribe - Run re-engagement campaigns - Improve  retention by 
 targeting at-risk users
