import streamlit as st

st.title("🤖 Vedic Ethics Robot Prototype")
st.write("Hello Mehak! This is your first Streamlit app running in the cloud.")

question = st.text_input("Ask me something:")
if question:
    st.success(f"You asked: {question}")
    st.info("Later, I’ll add the ethical reasoning engine here.")
