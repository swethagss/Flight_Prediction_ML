# ✈️ Flight Price Prediction

## 📄 Overview

This project focuses on building a predictive model to estimate flight prices based on various features such as the search date, departure date, airline, number of stops, and others. The goal is to provide insights into the factors influencing flight prices and create a reliable model for predicting future prices.

## 📑 Table of Contents

- 📚 [Libraries and Dataset](#libraries-and-dataset)
- ⚙️ [Data Preprocessing](#data-preprocessing)
- 📊 [Exploratory Data Analysis](#exploratory-data-analysis)
- 🏋️‍♂️ [Model Training & Selection](#model-training--selection)
- 📈 [Model Evaluation](#model-evaluation)
- 🚀 [Deployment](#deployment)
-  💻 [Run Instructions](#run-instructions)
- 📝 [Conclusion](#conclusion)

The following libraries were used in this project:

- **Pandas 🐼**: For handling and analyzing the data.
- **NumPy 📐**: For performing numerical operations.
- **Matplotlib & Seaborn 📊**: For visualizing the data.
- **Scikit-learn 🤖**: For building, evaluating, and deploying the models.

The dataset contains details about flight searches and their corresponding prices. The main features in the dataset include:

- **Searched Date**: The date the flight was searched.
- **Departure Date**: The date the flight departs.
- **Arrival Date**: The expected arrival date of the flight.
- **Flight Lands Next Day**: A flag indicating if the flight lands the next day.
- **Departure Airport**: The airport code or name from where the flight departs.
- **Arrival Airport**: The airport code or name where the flight arrives.
- **Number Of Stops**: The number of stops between departure and arrival.
- **Route**: The sequence of stops between departure and arrival.
- **Airline**: The airline operating the flight.
- **Cabin**: The class of the cabin (e.g., Economy, Business).
- **Price**: The price of the flight ticket.

## ⚙️ Data Preprocessing

The preprocessing steps included:

- **Handling Missing Values 🧩**: Filling in missing values to ensure data completeness.
- **Feature Engineering 🔧**: Creating additional features that could provide better predictive power.
- **Encoding Categorical Variables 🔢**: Converting categorical variables into numerical values for model use.
- **Splitting the Data ✂️**: Dividing the data into training and test sets for model evaluation.

## 📊 Exploratory Data Analysis (EDA)

EDA was conducted to understand the distribution and relationships between the features. Key visualizations and insights were:

- **Price Distribution 📈**: Analyzing how flight prices vary across different airlines, cabin classes, and other factors.
- **Correlation Analysis 🔗**: Investigating how the features relate to the target variable, Price.

## 🏋️‍♂️ Model Training & Selection

Several regression models were tested to predict flight prices:

- **Linear Regression 📊**: A simple, interpretable model that assumes a linear relationship between input features and the target variable. It’s fast to train but may underperform when data is highly non-linear or contains many outliers.
- **Ridge Regressor 🌉**: A linear regression variant with regularization to avoid overfitting by penalizing large coefficients.
- **Lasso Regressor 🧬**: Another regularized linear model that helps in feature selection by driving unimportant features' coefficients to zero.
- **Gradient Boosting Regressor 🌱**: An ensemble method that builds several weak learners sequentially, each correcting errors from the previous one.
- **XGBoost Regressor 🚀**: A faster and more efficient implementation of gradient boosting, often used in competitive machine learning tasks.
- **Random Forest Regressor 🌳**: An ensemble of decision trees trained on random subsets of data and features. The final prediction is the average of all the trees, making it robust against overfitting and capturing complex feature interactions.
- **LightGBM 🌟**: A gradient boosting framework that is highly efficient and scalable, designed to handle large datasets quickly by using histogram-based algorithms. It’s known for its speed and low memory usage.
- **CatBoost 🐱**: A gradient boosting algorithm developed by Yandex, optimized for categorical feature handling and less prone to overfitting. It automatically handles categorical variables without requiring explicit encoding.
- **Gradient Boosting Regressor 🌱**: An ensemble method that builds several weak learners sequentially, each correcting errors from the previous one. This approach often leads to high accuracy but can be slow to train.
- **AdaBoost Regressor ⚡**: An ensemble technique that adjusts the weights of weak learners based on their previous performance. AdaBoost focuses more on the mistakes made by the prior models, leading to improved performance by combining several weak learners into a strong one.


### 🥇 Best Model Selection

After evaluating all models, the **Random Forest Regressor** was chosen as the best performer. It demonstrated superior accuracy and could handle a wide range of data distributions and feature interactions. The Random Forest model outperformed the others in key metrics such as **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, **R-Squared (R²)** and **Adjusted R-Squared (R²)**.

This model was selected for its ability to generalize well on unseen data while remaining stable and robust in different scenarios.

## 📈 Model Evaluation

The models were assessed using the following metrics:

- **Mean Absolute Error (MAE) 📏**
- **Mean Squared Error (MSE) 🧮**
- **Root Mean Squared Error (RMSE) 🧮**
- **R-Squared (R²) 🔢**
- **Adjusted R-Squared (R²) 🔢**

The **Random Forest Regressor** was found to be the best-performing model based on these evaluation metrics.

<img width="683" alt="Screenshot 2024-12-19 at 11 54 20 AM" src="https://github.com/user-attachments/assets/641ba7bc-9ff9-48d0-8f5e-1a81a5228d49" />

### 🔍 Insights

- Non-linear models like **Random Forest** and **XGBoost** significantly outperformed linear models such as **Ridge** and **Lasso Regression**.
- **Random Forest Regressor** achieved an R² score of **0.7460**, which is a strong indication of its ability to capture complex relationships between the features and price.
- **XGBoost Regressor & CatBoost Regressor** also showed strong performance due to its ability to handle missing data and prevent overfitting through regularization.

### 🧩 Feature Importance

One key advantage of the **Random Forest** model is its ability to highlight the importance of different features in predicting flight prices. A feature importance plot was created to show which features had the most influence on price prediction.

### 🔍 Insights

- **Travel Time** was identified as the most important feature, followed by **Airline** and **Number of Stops**.
- Features like **Cabin** and **Flight Lands Next Day** had less influence on the prediction, indicating their minimal impact on price estimation.

### 🚀 Deployment

Deploy the model using a Streamlit app (app.py). The app allows users to input flight price prediction data and get price predictions.

https://github.com/user-attachments/assets/f21cf678-d383-4106-ab67-e054a2221655

### 📝 Conclusion

The project successfully demonstrated the ability to predict flight prices using machine learning techniques. The insights gained from the analysis can be valuable for airlines and consumers alike in understanding the dynamics of flight pricing.


