import streamlit as st
import pandas as pd
import joblib

# load the trained model
saved_model = joblib.load('flight_fare_model.pkl')

model = saved_model['model']
model_columns = saved_model['columns']

# app title
st.title('✈️ Flight Fare Prediction')
st.write('Enter your flight details to predict the ticket price.')

# user inputs
airline = st.selectbox('airline', [
    'AirAsia', 'Air_India', 'GO_FIRST',
    'Indigo', 'SpiceJet', 'Vistara'
])

source_city = st.selectbox('source city', [
    'Bangalore', 'Chennai', 'Delhi',
    'Hyderabad', 'Kolkata', 'Mumbai'
])

destination_city = st.selectbox('destination city', [
    'Bangalore', 'Chennai', 'Delhi',
    'Hyderabad', 'Kolkata', 'Mumbai'
])

departure_time = st.selectbox('departure time', [
    'Afternoon', 'Early_Morning', 'Evening',
    'Late_Night', 'Morning', 'Night'
])

arrival_time = st.selectbox('arrival time', [
    'Afternoon', 'Early_Morning', 'Evening',
    'Late_Night', 'Morning', 'Night'
])

stops = st.selectbox('number of stops', [
    'zero', 'one', 'two_or_more'
])

flight_class = st.selectbox('class', [
    'Economy', 'Business'
])

duration = st.number_input(
    'flight duration (hours)',
    min_value=0.83,
    max_value=49.83,
    value=2.5
)

days_left = st.number_input(
    'days left before departure',
    min_value=1,
    max_value=49,
    value=10
)

# prediction button
if st.button('predict fare'):

    input_data = pd.DataFrame([{
        'airline': airline,
        'source_city': source_city,
        'destination_city': destination_city,
        'departure_time': departure_time,
        'arrival_time': arrival_time,
        'stops': stops,
        'class': flight_class,
        'duration': duration,
        'days_left': days_left
    }])

    # encode categorical inputs
    input_encoded = pd.get_dummies(input_data, dtype=int)

    # match training columns
    input_encoded = input_encoded.reindex(
        columns=model_columns,
        fill_value=0
    )

    # predict ticket price
    prediction = model.predict(input_encoded)[0]

    st.success(f'Estimated ticket price: ₹{prediction:,.2f}')