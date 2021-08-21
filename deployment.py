''' Car Price Prediction Deployment '''

from flask import Flask, render_template,request
import pickle

app = Flask(__name__)

model = pickle.load(open("D:/Studies/Course Material/car_price_prediction_project/lasso_regressor.pkl","rb"))

@app.route("/",methods=['GET'])
def home():
    return render_template('predict.html')

@app.route("/predict",methods=['POST'])
def price():

    if request.method == 'POST':
        Present_Price = float(request.form['present_price'])
        year = int(request.form['year'])
        Age_Car = 2021-year
        Kms_Driven = int(request.form['kms'])
        Owner = int(request.form['owners'])
        fuel_type = request.form['fuel_type']
        if fuel_type == 'Petrol':
            Fuel_Type_Petrol = 1
            Fuel_Type_Diesel = 0
            Fuel_Type_CNG = 0
        elif fuel_type == 'Diesel':
            Fuel_Type_Petrol = 0
            Fuel_Type_Diesel = 1
            Fuel_Type_CNG = 0
        else:
            Fuel_Type_Petrol = 0
            Fuel_Type_Diesel = 0
            Fuel_Type_CNG = 1
        seller = request.form['seller']
        if seller == 'individual':
            Seller_Type_Dealer = 0
            Seller_type_Individual = 1
        else:
            Seller_Type_Dealer = 1
            Seller_type_Individual = 0
        transmission = request.form['Transmission']
        if transmission == 'manual':
            Transmission_Automatic = 0
            Transmission_Manual = 1
        else:
            Transmission_Automatic = 1
            Transmission_Manual = 0
        prediction = model.predict([[Present_Price,Kms_Driven,Owner,Age_Car,Fuel_Type_CNG,Fuel_Type_Diesel,
                                     Fuel_Type_Petrol,Seller_Type_Dealer,Seller_type_Individual,
                                     Transmission_Automatic,Transmission_Manual]])
        output = round(prediction[0],2)
        
        if output<0:
            return render_template("predict.html", prediction_text = "You can't sell this car")
        else:
            return render_template("predict.html", prediction_text = "You can sell this car at {} Lakhs INR".format(output))
    else:
        return render_template("predict.html")
        
if __name__ == '__main__':
    app.run()

