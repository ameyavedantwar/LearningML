import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("D:/6) work/Upskill/Python Kaggle/NC/train.csv")
#print(df.head())

df=df.dropna()
#print(df.isnull().sum())


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
#df['Material_encoded', 'Size_encoded','Laptop Compartment_encoded','Waterproof_encoded','Style_encoded','Color_encoded'] = label_encoder.fit_transform(df['Material', 'Size','Laptop Compartment','Waterproof','Style','Color'])

df['Material_encoded'] = label_encoder.fit_transform(df['Material'])
df['Size_encoded'] = label_encoder.fit_transform(df['Size'])
df['Laptop Compartment_encoded'] = label_encoder.fit_transform(df['Laptop Compartment'])
df['Waterproof_encoded'] = label_encoder.fit_transform(df['Waterproof'])
df['Style_encoded'] = label_encoder.fit_transform(df['Style'])
df['Color_encoded'] = label_encoder.fit_transform(df['Color'])

df = df.drop(columns=['Material','Size','Laptop Compartment','Waterproof','Style','Color'])
print(df)

#Create the heatmap with better color schemes and granularity
correlation_matrix = df.corr()
correlation_matrix_rounded = correlation_matrix.round(3)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix_rounded, annot=True, cmap='coolwarm', center=0, fmt='.3f', cbar_kws={'shrink': 0.8})



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Define independent variables (X) and dependent variables (Y)
X = df[['Material_encoded', 'Size_encoded','Laptop Compartment_encoded','Waterproof_encoded','Style_encoded','Color_encoded', 'Compartments','Weight Capacity (kg)' ]]   # Independent variables
Y = df[['Price']]  # Dependent variables

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create the model and train it
model = LinearRegression()
model.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = model.predict(X_test)

import pickle
# 1. Save the model
with open('linear_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# 2. Later, load the model
with open('linear_regression_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

y_test = pd.read_csv('D:/6) work/Upskill/Python Kaggle/NC/test.csv')
# Make predictions using the loaded model
#X_test = [[5], [6]]
predictions = loaded_model.predict(y_test)
print(predictions)

'''
# Evaluate the model for each target variable
mse_y1 = mean_squared_error(Y_test['Y1'], Y_pred[:, 0])
r2_y1 = r2_score(Y_test['Y1'], Y_pred[:, 0])

mse_y2 = mean_squared_error(Y_test['Y2'], Y_pred[:, 1])
r2_y2 = r2_score(Y_test['Y2'], Y_pred[:, 1])

# Print results for each dependent variable
print(f"Mean Squared Error (Y1): {mse_y1}")
print(f"R-squared (Y1): {r2_y1}")

print(f"Mean Squared Error (Y2): {mse_y2}")
print(f"R-squared (Y2): {r2_y2}")

# Model coefficients and intercepts for each dependent variable
print(f"Coefficients for Y1: {model.coef_[0]}")
print(f"Intercept for Y1: {model.intercept_[0]}")

print(f"Coefficients for Y2: {model.coef_[1]}")
print(f"Intercept for Y2: {model.intercept_[1]}")
'''
