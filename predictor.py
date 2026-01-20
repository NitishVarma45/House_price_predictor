import joblib
import pandas as pd

# Loading trained model
model = joblib.load("house_price_model.pkl")

def predict_price(location, total_sqft, bath, bhk):
    input_df = pd.DataFrame([{
        'location': location,
        'total_sqft': total_sqft,
        'bath': bath,
        'bhk': bhk
    }])

    price = model.predict(input_df)[0]
    return round(price, 2)

#input
location = input("Enter location: ")
total_sqft = float(input("Enter total sqft: "))
bath = int(input("Enter number of bathrooms: "))
bhk = int(input("Enter BHK: "))

predicted_price = predict_price(location, total_sqft, bath, bhk)

print(f"\nEstimated House Price: {predicted_price} Lakhs")
