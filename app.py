#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 22:03:00 2024

@author: swetha
"""

import streamlit as st
import pandas as pd
import joblib 

## Set the title with emoji
st.title("✈️ Flight Price Prediction")

# Load the model and scaler
#model_dict = joblib.load(''/Users/swetha/Desktop/My files/Projects/flight price prediction/fltprcprediction/model_data.joblib')
model_dict = joblib.load('fltprcprediction/model_data.joblib')
model = model_dict['model']
scaler = model_dict['scaler']

# Mappings for categorical variables
airline_mapping = {
    'Alaska Airlines':1,
    'American Airlines':2,
    'Delta' :3,
    'Spirit Airlines':4,
    'United Airlines':5
}

airport_mapping = {
    'LAS': 1,
    'LAX': 2,
    'SFO': 3,
    'BOS': 4,
    'JFK': 5,
    'ORD': 6
}

cabin_mapping = {
    'Economy' : 1,
    'Business' :2,
    'First' :3}

## Input fields for the user to enter flight details
st.header("Enter Flight details")

flight_lands_next_day = st.selectbox('Does the flight land next day ?🌙', ['Yes', 'No'])

# Map the user's selection to the corresponding numerical value
flight_lands_next_day = 1 if flight_lands_next_day == 'Yes' else 0

departure_airport = airport_mapping[st.selectbox('Departure Airport 🛫', airport_mapping.keys())]

arrival_airport = airport_mapping[st.selectbox('Arrival Airport 🛬', airport_mapping.keys())]

number_of_stops = st.number_input('Number of Stops ⛔️', min_value=0, step=1)
airline = airline_mapping[st.selectbox('Airline 🛩️', airline_mapping.keys())]
cabin = cabin_mapping[st.selectbox('Cabin Class 🎟️', cabin_mapping.keys())]
days_before_travel = st.number_input('Days Before Travel 📅', min_value=0, step=1)
travel_time = st.number_input('Travel Time (in hours) ⏰', min_value=0.0, step=0.1)

# Prediction

input_data = pd.DataFrame({
    'Flight Lands Next Day': [flight_lands_next_day],
    'Departure Airport' :[departure_airport],
    'Arrival Airport' :[arrival_airport],
    'Number Of Stops':[number_of_stops],
    'Airline': [airline],
    'Cabin': [cabin],
    'DaysbeforeTravel':[days_before_travel],
    'TravelTime':[travel_time]
    })

#scale the inputs using the loaded scaler
input_data_scaled = scaler.transform(input_data)

# Predict the price 
prediction = model.predict(input_data_scaled)
if st.button('Predict'):
    st.write(f"Predicted Flight price: ${prediction[0]:.2f}")





























