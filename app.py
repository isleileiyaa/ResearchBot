import streamlit as st
from agent import build_graph

st.title("ResearchBot — 智能研究助手")
st.write("输入问题，自动判断从企业知识库或网络搜索获取答案")
st.divider()

if "graph" not in st.session_state:
    st.session_state["graph"] = build_graph()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 显示历史消息
for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input("请输入你的问题...")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.spinner("Agent思考中..."):
        result = st.session_state["graph"].invoke({
            "question": prompt,
            "rag_result": "",
            "search_result": "",
            "final_answer": "",
            "route": ""
        })

        # 显示路由决策
        route = result["route"]
        source = "企业知识库" if route == "rag" else "网络搜索"
        st.info(f"信息来源：{source}")

        answer = result["final_answer"]
        st.chat_message("assistant").write(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})