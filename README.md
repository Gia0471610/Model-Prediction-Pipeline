<h1 align="center">Model-Prediction-Pipeline</h1>

## :star: OVERVIEW
1. Using PyCaret for classification tasks, from setting up the environment, training and evaluating models, making predictions, and visualizing model performance.
2. Using scikit-learn to build and evaluate a logistic regression model using a pipeline. 
3. Using Yellowbrick to performs model evaluation and prediction pipeline.


## :star: DATA
The data is from Kaggle "GooglePlayStore.csv", which is a dataset that contains information about mobile apps available on the Google Play Store. The dataset typically includes information such as app name, category, rating, number of reviews, size, installs, type (free or paid), price, content rating, genre, last updated date, and current version of the app.

## :star: Technologies Used
- :gear: Pycaret3
- :chart_with_upwards_trend: Scikit-learn
- :bar_chart: Matplotlib
- :bulb: Yellowbrick

## :star: Work Process
1. Setup --> Model Training and Selection --> Model Evaluation --> Predictions on Hold-out/Test Set --> Predictions on New Data --> Model Visualization
2. Data Preparation --> Data Solitting --> Pipeline Creation --> Model Training --> Model Prediction --> Model Evaluation
3. Split the dataset into training and testing sets --> Define a list of machine learning models to be evaluated --> Define a function 'score_model' --> Loop through each model to evaluate each model and store the accuracy scores --> Create a bar chart to compare the performance of different models based on accuracy.

## :star: Conclusion
Based on these results, it can be concluded that RandomForestClassifier performed the best among the four classifiers, achieving the highest accuracy, precision, and F1 score of 100%. AdaBoostClassifier and XGBClassifier also performed reasonably well, while GradientBoostingClassifier had comparatively lower performance in terms of accuracy and F1 score. 

Furthermore, our analysis indicates that the "Reviews" feature plays a key role in determining app success. The high importance value assigned to the "Reviews" feature suggests that the number of reviews an app has in the Google Play Store is a critical factor in predicting its success or failure in terms of installations. A higher number of reviews may indicate increased user engagement, satisfaction, or popularity, which can positively impact the app's success in terms of installations. 
