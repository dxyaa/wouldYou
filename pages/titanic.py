import streamlit as st
import pandas as pd
import numpy as np
from adaline_titanic import predict_new_passenger

st.write("# **WouldYou** survive the **Titanic**?🚢")

name = st.text_input("Enter your name")
age = st.number_input('Enter your age', max_value=100, min_value=0)
sex = st.selectbox("Select sex", ("male", "female"))


st.markdown(f""" 
### Passenger details:
- **Name:** {name}  
- **Age:** {age}  
- **Sex:** {sex}  
""")


new_passenger = [1001,1,sex,age,0,0,25,"Unknown","S"]
st.button("WouldYou Live ❓", on_click=lambda: st.session_state.update({"result": predict_new_passenger(new_passenger)}))

if "result" in st.session_state:
    st.write(st.session_state["result"])


result = predict_new_passenger(new_passenger)
st.write("# You have survived ✅" if result == "Survived" else "# You did not survive ❌")

