import streamlit as st
import time
if st.button("Run Detection"):

    st.write("Client Connected...")
    time.sleep(1)

    st.write("Sending parameters...")
    time.sleep(1)

    st.write("Aggregation running...")
    time.sleep(2)

    st.success("Prediction: SPAM")

st.title("Federated Spam Detection Demo")

message = st.text_input("Enter Message")

if st.button("Run Detection"):
    st.write("Client Connected...")
    st.write("Sending parameters...")
    st.write("Aggregation running...")
    st.success("Prediction: SPAM")