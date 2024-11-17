from flask import Flask, render_template,request, send_file
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import math

app = Flask(__name__)

# Load the trained model and feature names
model, feature_names = joblib.load('spare_parts_predictor.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    # Load the dataset
    file_path = "Online Retail.xlsx"
    data = pd.read_excel(file_path)
    
    # Preprocess the data to match training
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data['Year'] = data['InvoiceDate'].dt.year
    data['Month'] = data['InvoiceDate'].dt.month
    data['Day'] = data['InvoiceDate'].dt.day
    data.dropna(subset=['CustomerID'], inplace=True)
    data = pd.get_dummies(data, columns=['StockCode', 'Country'], drop_first=True)
    
    # Align columns with training features
    X_test = data[feature_names]
    y_test = data['Quantity']  # Assuming 'Quantity' is the target column in the dataset

    # Ensure alignment between predicted and actual values
    if len(y_test) > len(X_test):
        y_test = y_test[:len(X_test)]  # Truncate if y_test is longer

    # Make predictions
    y_pred = model.predict(X_test)

    # Plot predictions and actual values
    indices = range(len(y_pred))  # Use consistent indices for both
    plt.figure(figsize=(12, 6))
    plt.scatter(indices, y_test, color='orange', label='Actual Quantity', alpha=0.6)
    plt.scatter(indices, y_pred, color='blue', label='Predicted Quantity', alpha=0.6)
    plt.title('Actual vs Predicted Quantity of Spare Parts')
    plt.xlabel('Index')
    plt.ylabel('Quantity')
    plt.legend()
    plt.grid(True)
    plt.savefig('static/plot.png')
    plt.close()
    

    predicted_quantity = None
    if request.method == 'POST':
        # Get user inputs
        customer_id = request.form.get('CustomerID')
        country = request.form.get('Country')
        stock_code = request.form.get('StockCode')
        unit_price = request.form.get('UnitPrice')

        # Process user inputs into a DataFrame matching the model features
        user_data = {
            'CustomerID': [float(customer_id) if customer_id else 0],
            'Country': [country],
            'StockCode': [stock_code],
            'UnitPrice': [float(unit_price) if unit_price else 0],
        }

        # One-hot encode categorical variables and align with training features
        input_df = pd.DataFrame(user_data)
        input_df = pd.get_dummies(input_df, columns=['StockCode', 'Country'])
        for col in feature_names:  # Ensure all features exist in input data
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]  # Align column order with training data

        # Predict quantity
        predicted_quantity = model.predict(input_df)[0]
        predicted_quantity = math.ceil(predicted_quantity)

    #  Calculate performance metrics
    # mae = mean_absolute_error(y_test, y_pred)
    # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # r2 = r2_score(y_test, y_pred)
    # error_percentage = (mae / y_test.mean()) * 100

    return render_template(
        'index.html',
         predicted_quantity=predicted_quantity
    )

@app.route('/plot')
def plot():
    return send_file('static/plot.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
