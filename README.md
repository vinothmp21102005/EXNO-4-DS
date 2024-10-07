# EXNO:4-DS
## Name : VINOTH M P
## Reg No : 212223240182
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
file_path = '/content/income.csv'
data = pd.read_csv(file_path)
data.head()
```
![image](https://github.com/user-attachments/assets/5a8f56af-3218-4bd6-94dd-00d646438b15)
```
data.info()
```
![image](https://github.com/user-attachments/assets/2dd79077-e180-4e23-b66d-a48e898a631a)
```
# One-hot encoding categorical columns
data_encoded = pd.get_dummies(data.drop('SalStat', axis=1), drop_first=True)

# Step 2: Apply Feature Scaling - StandardScaler, MinMaxScaler, etc.

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

# Extract numerical columns for scaling
numerical_cols = ['age', 'capitalgain', 'capitalloss', 'hoursperweek']

# StandardScaler
standard_scaler = StandardScaler()
data_standard_scaled = data_encoded.copy()
data_standard_scaled[numerical_cols] = standard_scaler.fit_transform(data_standard_scaled[numerical_cols])
data_standard_scaled[numerical_cols] 
```
![image](https://github.com/user-attachments/assets/0c89891e-b279-4bfa-9f68-2782fa340190)

```
# MinMaxScaler
min_max_scaler = MinMaxScaler()
data_min_max_scaled = data_encoded.copy()
data_min_max_scaled[numerical_cols] = min_max_scaler.fit_transform(data_min_max_scaled[numerical_cols])
data_min_max_scaled[numerical_cols] 
```
![image](https://github.com/user-attachments/assets/0b436a4a-1dd0-405e-8da0-2b8f8bcbbffc)

```

# Maximum Absolute Scaling
max_abs_scaler = MaxAbsScaler()
data_max_abs_scaled = data_encoded.copy()
data_max_abs_scaled[numerical_cols] = max_abs_scaler.fit_transform(data_max_abs_scaled[numerical_cols])

data_max_abs_scaled[numerical_cols]
```
![image](https://github.com/user-attachments/assets/8e6d494d-0d71-4cf2-9c38-b509328caa48)

```
# RobustScaler
robust_scaler = RobustScaler()
data_robust_scaled = data_encoded.copy()
data_robust_scaled[numerical_cols] = robust_scaler.fit_transform(data_robust_scaled[numerical_cols])

data_robust_scaled[numerical_cols] 
```
![image](https://github.com/user-attachments/assets/561fb862-f489-43a2-baa4-45949f41d3cd)
```
# Step 3: Apply Feature Selection - Using SelectKBest with ANOVA F-test (for classification)

from sklearn.feature_selection import SelectKBest, f_classif
y = data['SalStat'].apply(lambda x: 1 if x == 'greater than 50,000' else 0) 
X = data_standard_scaled
selector = SelectKBest(score_func=f_classif, k=10)  # Select top 10 features
X_new = selector.fit_transform(X, y)
selected_feature_names = X.columns[selector.get_support()]
selected_features = pd.DataFrame(X_new, columns=selected_feature_names)
selected_feature_names
```
![image](https://github.com/user-attachments/assets/f5137c26-198b-4a9f-b1c8-0d9513922a20)
```
# Save the selected features to a CSV file
output_file = '/mnt/data/selected_features.csv'
selected_features.to_csv(output_file, index=False)
output_file
```

# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is successfully written and executed
