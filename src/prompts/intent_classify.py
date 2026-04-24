"""
意图分类 Prompt 模板。

定义合法意图枚举集合（VALID_INTENTS）和意图分类提示词模板。
意图枚举固定，禁止在此文件外扩展。
"""

# 合法意图枚举（固定，禁止在此文件外扩展）
VALID_INTENTS = frozenset(
    {"rag_query", "paper_search", "paper_download", "build_knowledge", "general_chat"}
)

INTENT_CLASSIFY_PROMPT = """你是一个意图分类助手。根据用户输入，判断用户意图。

可选意图（只能返回以下之一）：
- rag_query：用户需要根据已建立的知识库回答问题（如问某篇论文的内容）
- paper_search：用户需要搜索 arXiv 上的论文
- paper_download：用户需要下载论文
- build_knowledge：用户需要构建知识库（对已下载论文建立向量索引）
- general_chat：通用对话，无需调用任何工具或知识库

当前会话情景（最近操作记录，辅助判断"这篇"指代的论文）：
{context}

对话历史（最近几轮，供参考）：
{history}

用户输入：{user_input}

请以严格的 JSON 格式返回，不要输出任何 JSON 以外的内容：
{{"intent": "...", "confidence": 0.95, "reasoning": "..."}}"""
