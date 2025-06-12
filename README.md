# Salary Category Prediction — ML Model Zoo
Welcome to the repository for my participation in a Machine Learning competition focused on employee salary category prediction.

This repo explores different machine learning approaches and ensemble methods to optimize prediction accuracy using a combination of categorical, numerical, and high-dimensional vector features.

Problem Statement
The goal is to predict the salary category of job postings using structured data (job title, state, date, etc.) and vectorized text features (job description embeddings). The target variable is salary_category, which we aim to predict with high precision.

Features Used
Structured categorical data: job titles, job states, posting dates
Boolean features: binary job attributes
Numerical features: preprocessed vector values in feature columns
High-dimensional vectors: job_desc_001 to job_desc_300 (text embedding vectors)
Engineered features: extracted year/month from date, missing value indicators, etc.

Model	Description	
LightGBM Baseline
Tuned LightGBM	Optimized hyperparameters + regularization
XGBoost	Gradient boosting with regularization	
CatBoost	Categorical boosting with internal encoding	
Random Forest	Ensemble of decision trees
Stacking Ensemble	Combined LGBM, XGB, CatBoost	
PCA + LightGBM	Dimensionality reduction before training	

Preprocessing Highlights
Handling of missing values (job_state, job_posted_date)
Feature encoding (LabelEncoder, Ordinal, custom mappings)
Standardization and SVD/PCA for dimensionality reduction
Balanced class weights to handle imbalance

