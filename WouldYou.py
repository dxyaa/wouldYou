import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="WouldYou",
    page_icon="❓",
)

st.write("# WouldYou❓🚀")

st.markdown(
    """
   This is **WouldYou**, the ultimate game of survival and absurdity!

Ever wondered if you'd make it out alive in a Titanic-style disaster? How about a road accident, a shipwreck, or the outcome of your favourite character battles? 🧟‍♂️ 

Well, wonder no more! We're here to answer the important questions in life!

**How It Works:**

🔹 Choose your WouldYou scenario.

🔹 Select your traits.

🔹 Hit the button and check out your destiny!

Would you? There's only one way to find out. 🚀🎭
"""
)
st.page_link("pages/titanic.py", label="Would you survive Titanic?", icon="🚢")

st.page_link("pages/titanic.py", label="Would you survive Estonia?", icon="🚤")

st.page_link("pages/titanic.py", label="Would you survive road accidents?", icon="🚑")

st.page_link("pages/titanic.py", label="Would you win a comic battle?", icon="⚔️")

