import streamlit as st
import pandas as pd
import numpy as np
from adaline_titanic import predict_new_passenger

st.write("# **WouldYou** survive the **Titanic**?🚢")

# Input fields
pid = st.number_input("Enter Passenger ID")
pclass = st.selectbox("Select Ticket Class", ("1", "2", "3"))
name = st.text_input("Enter your name")
sex = st.selectbox("Select sex", ("male", "female"))
age = st.number_input("Enter your age", max_value=100, min_value=0)
sibsp = st.number_input("Number of siblings/spouses", min_value=0)
parch = st.number_input("Number of parents and children", min_value=0)
ticketno = st.number_input("Ticket Number", min_value=0)
fare = st.number_input("Ticket fare paid", min_value=0)
cabin = st.text_input("Cabin ID")
embarked = st.selectbox("Select Embarked Class", ("S", "Q", "C"))

# Display passenger details
st.markdown(f""" 
### Passenger details:
- **Passenger Id:** {pid}
- **Name:** {name}  
- **Age:** {age}  
- **Sex:** {sex}
- **Siblings/Spouses:** {sibsp}
- **Parents/Children:** {parch}
- **Ticket number:** {ticketno}
- **Fare:** {fare}
- **Cabin ID:** {cabin}
- **Embarked Class:** {embarked}
""")

# Create the new passenger data
new_passenger = [pid, int(pclass), sex, age, sibsp, parch, fare, cabin, embarked]

# Button to trigger prediction
if st.button("WouldYou Live ❓"):
    # Call the prediction function and store the result in session state
    result = predict_new_passenger(new_passenger)
    st.session_state["result"] = result

# Display the result if it exists in session state
if "result" in st.session_state:
    st.write("# You have survived ✅" if st.session_state["result"] == "Survived" else "# You did not survive ❌")