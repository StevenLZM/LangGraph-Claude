import streamlit as st

st.write("开始执行")

name = st.text_input("输入名字")

st.write("你好", name)


if "count" not in st.session_state:
    st.session_state.count = 0

if st.button("点击"):
    st.session_state.count += 1

st.write(st.session_state.count)
st.write(str(st.session_state.to_dict()))