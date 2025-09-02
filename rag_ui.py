import streamlit as st
import requests

st.title("AA Assistant - Document Query")

query = st.text_input("Enter your question: ")

if st.button("Submit") and query:
    response = requests.post("http://127.0.0.1:8000/query", json={"query": query})
    if response.status_code == 200:
        data = response.json()
        st.subheader("Answer: ")
        st.write(data["answer"])
        st.subheader("Sources: ")
        for src in data["source"]:
            st.write(f"- {src}")
    else:
        st.error("Failed to get response from API")