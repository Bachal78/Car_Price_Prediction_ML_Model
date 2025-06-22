import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#load the dataset
df = pd.read_csv(r'dataset.csv')

#convert categorical variables(features) to numerical values
df["Fuel_Type"] = df['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2})
df["Transmission"] = df['Transmission'].map({'Manual': 0, 'Automatic': 1})
df['Seller_Type'] = df['Seller_Type'].map({ 'Dealer': 0, 'Individual': 1})

#encode a car name
car_name_encoder = LabelEncoder()
df['Car_Name'] = car_name_encoder.fit_transform(df['Car_Name'])


#features and target(label) value
X = df[['Car_Name', 'Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Transmission', 'Seller_Type']]
y = df['Selling_Price']

#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create and train the model
model = LinearRegression()
model.fit(X_train, y_train) #fit means training the model

#predict on the test data
y_pred = model.predict(X_test)

#calculate RMSE and R^2 score to understand the model's accuracy
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R^2 Score:", r2)

#Calculate the residuals as the difference between y_test and y_pred
residuals = y_test - y_pred

#Visual Representation
# matplotlib.use('TKAgg')
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals, color='blue', edgecolor='w', size=70)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.savefig('residuals_plot.png')
print("Residuals plot saved as 'residuals_plot.png'. Check your working directory.")


# #Streamlit app
st.title("Car Price Prediction")
st.write('Enter the details to predict the price of used car: ')

car_name= st.selectbox("Car Name", options=car_name_encoder.classes_)
year = st.number_input("Year of Car", min_value=2000, max_value=2025, value=2015)
present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, max_value=50.0, value=5.0)
kms_driven = st.number_input("Kms Driven", min_value=0, max_value=300000, value=30000)
fuel_type = st.selectbox("Fuel Type", options=['Petrol', 'Diesel', 'CNG'])
transmission = st.selectbox("Transmission Type", options=['Manual', 'Automatic'])
owner = st.selectbox("Ownership Status", options=[0, 1, 2, 3])

# Convert inputs to numerical values
fuel_type_num = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}[fuel_type]
transmission = {'Manual': 0, 'Automatic': 1}[transmission]
car_name_encoded = car_name_encoder.transform([car_name])[0]

# Prepare the input data for prediction
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        'Car_Name': [car_name_encoded],
        'Year': [year],
        'Present_Price': [present_price],
        'Kms_Driven': [kms_driven],
        'Fuel_Type': [fuel_type_num],
        'Transmission': [transmission],
        'Seller_Type': [owner]
    })

    # Make prediction
    predicted_price = model.predict(input_data)[0]
    
    # Display the predicted price
    st.success(f"The predicted price of the car is: Rs {predicted_price:.2f} lakhs")
    # Display the residuals plot
    st.image('residuals_plot.png', use_column_width=True)
