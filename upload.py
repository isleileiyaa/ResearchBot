from knowledge_base import KnowledgeBaseService

service = KnowledgeBaseService()

# 上传企业内部文档
with open("../AI大模型RAG与agent开发/P4_RAG项目案例/data/企业内部文档.txt", "r", encoding="utf-8") as f:
    text = f.read()

result = service.upload_by_str(text, "企业内部文档.txt")
print(result)