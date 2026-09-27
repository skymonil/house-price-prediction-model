import pandas as pd


def predict_price(model, bedrooms, area, location, age):
    new_house = pd.DataFrame(
        {
            "Bedrooms": [bedrooms],
            "Area": [area],
            "Location": [location],
            "Age": [age],
        }
    )

    prediction = model.predict(new_house)

    return prediction[0]