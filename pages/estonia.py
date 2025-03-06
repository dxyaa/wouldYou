import streamlit as st
import pandas as pd
import numpy as np
from mp_estonia import predict_new_passenger

st.write("# **WouldYou** survive the **Estonia Disaster**?🚢")

pid = st.number_input("Enter Passenger ID",min_value=1,max_value=1000)
country = st.select_slider("Select your Country",("Sweden","Estonia","Russia","Morocco","Finland","Great Britain","Latvia","France"))
name = st.text_input("Enter your name")
age = st.number_input("Enter Age",min_value=1,max_value=100)
sex = st.selectbox("Select Sex",("M","F"))
category = st.selectbox("Select Category",("P","C"))


st.markdown(f"""
- **Passsenger Id:** {pid}
- **Name:** {name}
- **Country:** {country}
- **Age:** {age}  
- **Sex:** {sex}
- **Category:** {category}
""")


new_passenger = [country,sex,age,category]
if st.button("WouldYou Live ❓"):
    result = predict_new_passenger(new_passenger)
    st.session_state["result"] = result
if "result" in st.session_state:
    st.write("# You have survived ✅" if st.session_state["result"] == "Survived" else "# You did not survive ❌")
    