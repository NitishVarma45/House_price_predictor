This project predicts house prices in Bengaluru using machine learning.

It involves data cleaning, feature engineering, and outlier removal using domain knowledge.

Regression models including Linear Regression, Lasso, and Ridge are trained using a unified preprocessing pipeline for accurate and consistent predictions.

Step 1: Run data cleaning  
This cleans the raw dataset and generates cleandata.csv.  
python cleaning.py

Step 2: Train the model  
This trains regression models and saves the final model as house_price_model.pkl.  
python Modelling.py

Step 3: Predict house price  
Enter custom inputs to predict the house price.  
python predictor.py
