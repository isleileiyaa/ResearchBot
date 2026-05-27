from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from vector_stores import VectorStoreService
import config_data as config

# ============ 第一部分：定义State ============
class ResearchState(TypedDict):
    question: str        # 用户的问题
    rag_result: str      # RAG检索结果
    search_result: str   # 网络搜索结果
    final_answer: str    # 最终答案
    route: str           # 路由决策：rag 或 search
    reflection: str      # 反思
    loop_count: int      # 循环计数
   
# ============ 初始化模型和检索器 ============
llm = ChatTongyi(model="qwen-max")

vector_service = VectorStoreService(
    embedding=DashScopeEmbeddings(model=config.embedding_model_name)
)
retriever = vector_service.get_retriever()

# ============ 第二部分：路由节点 ============
def route_node(state: ResearchState) -> ResearchState:
    """判断用户问题应该走RAG还是网络搜索"""
    
    question = state["question"]
    
    prompt = f"""判断以下问题是否能从企业内部文档知识库中找到答案。
企业知识库包含：公司制度、员工手册、IT流程、HR政策等内部文档。

问题：{question}

如果问题关于企业内部事务，回答：rag
如果问题需要外部信息或知识库不太可能有答案，回答：search

只回答rag或search，不要说其他任何内容。"""

    response = llm.invoke(prompt)
    route = response.content.strip().lower()
    
    # 确保route只有rag或search两个值
    if "rag" in route:
        route = "rag"
    else:
        route = "search"
    
    print(f"路由决策：{route}")
    return {"route": route}
# ============ 第三部分：RAG检索节点 ============
def rag_node(state: ResearchState) -> ResearchState:
    """从企业知识库检索相关内容"""
    
    question = state["question"]
    
    # HyDE：先生成假答案再检索
    hyde_prompt = f"""请根据以下问题，生成一段可能出现在企业内部文档中的回答。
语言风格要正式，像企业制度文档。

问题：{question}

回答："""
    
    hyde_response = llm.invoke(hyde_prompt)
    hypothetical_answer = hyde_response.content
    
    # 用假答案检索
    docs = retriever.invoke(hypothetical_answer)
    
    if not docs:
        rag_result = "知识库中未找到相关内容"
    else:
        rag_result = "\n".join([doc.page_content for doc in docs])
    
    print(f"RAG检索结果：{rag_result[:100]}...")
    return {"rag_result": rag_result}


# ============ 第四部分：网络搜索节点 ============
def search_node(state: ResearchState) -> ResearchState:
    """使用Qwen联网搜索"""
    
    question = state["question"]
    
    # 开启Qwen的联网搜索功能
    from langchain_community.chat_models.tongyi import ChatTongyi
    search_llm = ChatTongyi(
        model="qwen3-max",
        model_kwargs={
            "enable_search": True  # 开启联网搜索
        }
    )
    
    prompt = f"请搜索并回答以下问题：{question}"
    response = search_llm.invoke(prompt)
    search_result = response.content
    
    print(f"网络搜索结果：{search_result[:100]}...")
    return {"search_result": search_result}
# ============ 测试路由节点 ============
# ============ 反思节点 ============
def reflect_node(state: ResearchState) -> ResearchState:
    """判断RAG检索结果是否足够回答问题"""
    
    question = state["question"]
    rag_result = state["rag_result"]
    loop_count = state.get("loop_count", 0)
    
    # 超过1次循环强制结束
    if loop_count >= 1:
        return {"reflection": "sufficient", "loop_count": loop_count}
    
    if rag_result == "知识库中未找到相关内容":
        print("反思：知识库无内容，需要联网搜索补充")
        return {"reflection": "insufficient", "loop_count": loop_count + 1}
    
    prompt = f"""判断以下检索结果是否足够回答用户问题。

用户问题：{question}
检索结果：{rag_result}

如果检索结果能完整回答问题，回答：sufficient
如果检索结果不够，需要补充外部信息，回答：insufficient

只回答sufficient或insufficient。"""

    response = llm.invoke(prompt)
    reflection = response.content.strip().lower()
    
    if "sufficient" in reflection:
        reflection = "sufficient"
    else:
        reflection = "insufficient"
    
    print(f"反思结果：{reflection}")
    return {"reflection": reflection, "loop_count": loop_count + 1}


def decide_reflection(state: ResearchState) -> str:
    return state["reflection"]
# ============ 第五部分：整合生成节点 ============
def generate_node(state: ResearchState) -> ResearchState:
    """整合检索结果，生成最终回答"""
    
    question = state["question"]
    rag_result = state["rag_result"]
    search_result = state["search_result"]
    
    # 根据有哪些内容来构造context
    context_parts = []
    
    if rag_result and rag_result != "知识库中未找到相关内容":
        context_parts.append(f"【企业内部文档】\n{rag_result}")
    
    if search_result:
        context_parts.append(f"【网络搜索结果】\n{search_result}")
    
    context = "\n\n".join(context_parts)
    
    
    prompt = f"""请根据以下参考资料，专业、简洁地回答用户问题。

参考资料：
{context}

用户问题：{question}

回答："""
    
    response = llm.invoke(prompt)
    final_answer = response.content
    
    return {"final_answer": final_answer}


# ============ 第六部分：条件路由函数 ============
def decide_route(state: ResearchState) -> str:
    """根据route字段决定下一个节点"""
    return state["route"]


# ============ 第七部分：组装LangGraph图 ============
def build_graph():
    graph = StateGraph(ResearchState)
    
    # 添加节点
    graph.add_node("route", route_node)
    graph.add_node("rag", rag_node)
    graph.add_node("search", search_node)
    graph.add_node("generate", generate_node)
    
    # 设置入口
    graph.set_entry_point("route")
    
    # 条件路由：route节点之后根据决策走rag或search
    graph.add_conditional_edges(
        "route",
        decide_route,
        {
            "rag": "rag",
            "search": "search"
        }
    )
    
    # rag和search之后都去generate
    graph.add_edge("rag", "generate")
    graph.add_edge("search", "generate")
    
    # generate之后结束
    graph.add_edge("generate", END)
    
    return graph.compile()


# ============ 测试 ============
if __name__ == '__main__':
    app = build_graph()
    
    # 测试问题一：应该走RAG
    print("="*30)
    print("问题一：年假怎么申请")
    result1 = app.invoke({
        "question": "年假怎么申请",
        "rag_result": "",
        "search_result": "",
        "final_answer": "",
        "route": ""
    })
    print(f"最终回答：{result1['final_answer']}")
    
    # 测试问题二：应该走搜索
    print("="*30)
    print("问题二：2026年最新的AI大模型有哪些")
    result2 = app.invoke({
        "question": "2026年最新的AI大模型有哪些",
        "rag_result": "",
        "search_result": "",
        "final_answer": "",
        "route": ""
    })
    print(f"最终回答：{result2['final_answer']}")