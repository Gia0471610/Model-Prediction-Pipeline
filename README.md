<h1 align="center">Model-Prediction-Pipeline</h1>

## OVERVIEW
1. Using PyCaret for classification tasks, from setting up the environment, training and evaluating models, making predictions, and visualizing model performance.
2. Using scikit-learn to build and evaluate a logistic regression model using a pipeline. 
3. Using Yellowbrick to performs model evaluation and prediction pipeline.


## DATA
The data is from Kaggle "GooglePlayStore.csv", which is a dataset that contains information about mobile apps available on the Google Play Store. The dataset typically includes information such as app name, category, rating, number of reviews, size, installs, type (free or paid), price, content rating, genre, last updated date, and current version of the app.

## Technologies Used
- :gear: Pycaret3
- :chart_with_upwards_trend: Scikit-learn
- :bar_chart: Matplotlib
- :bulb: Yellowbrick

## Work Process
1. Setup --> Model Training and Selection --> Model Evaluation --> Predictions on Hold-out/Test Set --> Predictions on New Data --> Model Visualization
2. Data Preparation --> Data Solitting --> Pipeline Creation --> Model Training --> Model Prediction --> Model Evaluation
3. Split the dataset into training and testing sets --> Define a list of machine learning models to be evaluated --> Define a function 'score_model' --> Loop through each model to evaluate each model and store the accuracy scores --> Create a bar chart to compare the performance of different models based on accuracy.
