"""
Paper Knowledge Base Q&A System - CLI Version
"""
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import asyncio
from datetime import datetime
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
PAPERS_ROOT = PROJECT_ROOT / "papers"
VECTOR_DB_ROOT = PROJECT_ROOT / "vector_db"
MEMORY_ROOT = PROJECT_ROOT / "memory"
SELECT_TOP_K = 3

load_dotenv(PROJECT_ROOT / ".env")
sys.path.append(str(PROJECT_ROOT))

from src.tools.download_tool import download_papers, PaperMetadata
from src.tools.parse_pdf_tool import parse_pdf
from src.tools.search_tool import arxiv_search_papers

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import json

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

def create_chat_model(model_name: str) -> ChatOpenAI:
    """
    Create a chat model instance.
    
    Args:
        model_name: Name of the model to use
    
    Returns:
        ChatOpenAI instance
    """
    return ChatOpenAI(
        model=model_name,
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
        temperature=0
    )


GLOBAL_MODEL = create_chat_model(os.getenv("MODEL_NAME", "gpt-3.5-turbo"))


class MemoryManager:
    """Session memory manager for persisting conversation history."""

    def __init__(self, storage_file: str = "paper_memory.json"):
        """
        Initialize memory manager.

        Args:
            storage_file: Name of the storage file
        """
        self.memory_dir = MEMORY_ROOT
        self.memory_dir.mkdir(exist_ok=True)
        self.storage_file = self.memory_dir / storage_file

    def save(self, session_id: str, messages: List[BaseMessage]) -> None:
        """
        Save conversation messages for a session.

        Args:
            session_id: Session identifier
            messages: List of messages to save
        """
        data = self._load()
        msg_list = [{"role": m.type, "content": m.content} for m in messages]
        data[session_id] = msg_list
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, session_id: str) -> List[BaseMessage]:
        """
        Load conversation messages for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of loaded messages
        """
        data = self._load()
        msg_list = data.get(session_id, [])
        return [
            HumanMessage(m["content"]) if m["role"] == "human" else AIMessage(m["content"])
            for m in msg_list
        ]

    def _load(self) -> Dict[str, Any]:
        """Load all memory data from file."""
        if not self.storage_file.exists():
            return {}
        with open(self.storage_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def list_all_sessions(self) -> List[str]:
        """
        List all session IDs.
        
        Returns:
            List of session IDs
        """
        data = self._load()
        return list(data.keys())
    
    def delete_session(self, session_id: str) -> None:
        """
        Delete a session from memory storage.
        
        Args:
            session_id: Session identifier to delete
        """
        data = self._load()
        if session_id in data:
            del data[session_id]
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    
    def rename_session(self, old_session_id: str, new_session_id: str) -> bool:
        """
        Rename a session in memory storage.
        
        Args:
            old_session_id: Old session identifier
            new_session_id: New session identifier
            
        Returns:
            True if rename successful
        """
        data = self._load()
        if old_session_id in data and new_session_id not in data:
            data[new_session_id] = data[old_session_id]
            del data[old_session_id]
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        return False


class PaperKnowledgeBase:
    """Paper knowledge base with vector storage and RAG capabilities."""

    def __init__(
        self,
        session_id: str,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize paper knowledge base.

        Args:
            session_id: Session identifier
            embedding_model: Name of the embedding model to use
        """
        self.session_id = session_id
        self.papers_dir = PAPERS_ROOT / session_id
        self.index_dir = VECTOR_DB_ROOT / session_id

        self.embeddings = OpenAIEmbeddings(
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("API_KEY"),
            model=embedding_model
        )

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""]
        )

        self.vector_store = None
        self.query_log = []

    def has_exist_knowledge(self) -> bool:
        """Check if knowledge base exists."""
        return self.index_dir.exists() and any(self.index_dir.iterdir())

    def list_exist_papers(self) -> List[str]:
        """List all PDF papers in the session directory."""
        if not self.papers_dir.exists():
            return []
        return [p.name for p in self.papers_dir.glob("*.pdf")]
    
    def _get_papers_info(self) -> str:
        """Get formatted information about papers in the knowledge base."""
        if not self.papers_dir.exists():
            return "暂无论文"
        
        papers = []
        for pdf_file in self.papers_dir.glob("*.pdf"):
            papers.append(f"- {pdf_file.name}")
        
        if not papers:
            return "暂无论文"
        
        return "\n".join(papers)

    def clear_old_knowledge(self) -> None:
        """Clear all knowledge base data and papers."""
        import shutil
        if self.papers_dir.exists():
            shutil.rmtree(self.papers_dir)
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
    
    def rename_folders(self, old_session_id: str, new_session_id: str) -> bool:
        """
        Rename papers and vector database folders.
        
        Args:
            old_session_id: Old session identifier
            new_session_id: New session identifier
            
        Returns:
            True if rename successful
        """
        import shutil
        old_papers_dir = PAPERS_ROOT / old_session_id
        new_papers_dir = PAPERS_ROOT / new_session_id
        old_index_dir = VECTOR_DB_ROOT / old_session_id
        new_index_dir = VECTOR_DB_ROOT / new_session_id
        
        success = True
        
        if old_papers_dir.exists():
            if not new_papers_dir.exists():
                shutil.move(str(old_papers_dir), str(new_papers_dir))
            else:
                success = False
        
        if old_index_dir.exists():
            if not new_index_dir.exists():
                shutil.move(str(old_index_dir), str(new_index_dir))
            else:
                success = False
        
        return success

    async def build_index(self, pdf_files: List[str]) -> bool:
        """
        Build vector index from PDF files.

        Args:
            pdf_files: List of PDF file paths

        Returns:
            True if index built successfully
        """
        all_chunks = []
        for pdf_path in pdf_files:
            parse_ret = await parse_pdf(pdf_path=pdf_path)
            for chunk in parse_ret["chunks"]:
                all_chunks.append({"chunk": chunk, "metadata": parse_ret["metadata"]})

        if not all_chunks:
            return False

        docs = [
            Document(page_content=c["chunk"], metadata=c["metadata"])
            for c in all_chunks
        ]
        self.vector_store = FAISS.from_documents(docs, self.embeddings)

        index_path = Path(self.index_dir)
        index_path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(index_path))
        return True

    def load_index(self) -> bool:
        """
        Load existing vector index.

        Returns:
            True if index loaded successfully

        Raises:
            Exception: If loading fails
        """
        try:
            self.vector_store = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return True
        except Exception as e:
            raise Exception(f"Failed to load vector index: {e}")

    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the knowledge base with RAG.
        
        Args:
            question: User question
            
        Returns:
            Query result with answer and sources
            
        Raises:
            Exception: If no vector index is loaded
        """
        if not self.vector_store:
            raise Exception("No vector index loaded, please load index first")

        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

        docs = retriever.invoke(question)
        
        papers_info = self._get_papers_info()

        prompt = ChatPromptTemplate.from_template("""
你是一个知识渊博的AI助手。请仅根据提供的上下文用中文回答问题。

上下文: {context}

当前知识库包含的论文:
{papers_info}

问题: {question}

说明:
1. 仅根据上下文回答，不要编造或使用外部信息
2. 如果上下文中没有相关信息，请这样回答：
   \"抱歉，当前知识库中没有找到与您问题相关的内容。当前知识库包含以下论文：
   {papers_info}
   如果您想了解其他主题，请先搜索并下载相关论文来更新知识库。\"
3. 提供详细和全面的答案
4. 适当时用项目符号清晰组织答案
5. 引用来源，包括论文标题、作者和年份（如果有）
""")

        def format_docs(documents):
            return "\n\n".join([doc.page_content for doc in documents])

        from langchain_core.runnables import RunnableLambda

        rag_chain = (
            {
                "context": RunnableLambda(lambda x: format_docs(retriever.invoke(x["question"]))),
                "papers_info": RunnableLambda(lambda x: papers_info),
                "question": RunnableLambda(lambda x: x["question"])
            }
            | prompt
            | GLOBAL_MODEL
            | StrOutputParser()
        )

        result = rag_chain.invoke({"question": question})

        sanitized_docs = []
        for doc in docs:
            sanitized_metadata = {}
            if "source" in doc.metadata:
                source_path = Path(doc.metadata["source"])
                sanitized_metadata["source"] = source_path.name
            for key, value in doc.metadata.items():
                if key != "source":
                    sanitized_metadata[key] = value
            sanitized_docs.append(sanitized_metadata)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.query_log.append({
            "question": question,
            "answer": result,
            "timestamp": timestamp,
            "docs": sanitized_docs
        })
        return {
            "question": question,
            "answer": result,
            "sources": docs,
            "source_count": len(docs),
            "timestamp": timestamp
        }


async def main():
    """Main CLI interface."""
    print("="*60)
    print("论文知识库问答系统")
    print("="*60)

    GLOBAL_SESSION_ID = input("请输入会话ID（默认default）：").strip() or "default"
    print(f"\n会话ID：{GLOBAL_SESSION_ID}")

    kb = PaperKnowledgeBase(GLOBAL_SESSION_ID)
    memory = MemoryManager()
    messages = memory.load(GLOBAL_SESSION_ID)

    has_knowledge = kb.has_exist_knowledge()
    has_papers = len(kb.list_exist_papers()) > 0

    if has_knowledge or has_papers:
        print(f"\n发现会话 {GLOBAL_SESSION_ID} 的已有资源：")
        if has_papers:
            print(f"  - 已下载论文：{kb.list_exist_papers()}")
        if has_knowledge:
            print(f"  - 知识库已构建")

        print("\n请选择操作：")
        print("1. 使用已有知识库进行问答")
        print("2. 重新构建知识库（保留下载文件）")
        print("3. 清除所有资源，重新搜索和下载")

        choice = input("请输入选择（1/2/3）：").strip()

        if choice == "1":
            if has_knowledge:
                kb.load_index()
                await query_mode(kb, memory, GLOBAL_SESSION_ID, messages)
                return
        elif choice == "2":
            pass
        elif choice == "3":
            kb.clear_old_knowledge()
        else:
            pass

    print("\n搜索参数")
    querys = input("请输入搜索关键词（逗号分隔）：").strip().split(",")
    querys = [q.strip() for q in querys if q.strip()]
    if not querys:
        print("请至少输入一个搜索关键词")
        return

    max_results = int(input("请输入最大结果数（默认50）：").strip() or "50")
    sort_by = input("请输入排序方式（Relevance/LastUpdatedDate/SubmittedDate，默认Relevance）：").strip() or "Relevance"
    sort_order = input("请输入排序顺序（Ascending/Descending，默认Descending）：").strip() or "Descending"
    start_date = input("请输入开始日期（格式：YYYY-MM-DD，可选）：").strip() or None
    end_date = input("请输入结束日期（格式：YYYY-MM-DD，可选）：").strip() or None

    print(f"\n正在搜索论文...")
    search_results = await arxiv_search_papers(
        querys=querys,
        max_results=max_results,
        sort_by=sort_by,
        sort_order=sort_order,
        start_date=start_date,
        end_date=end_date
    )

    if not search_results:
        print("未找到论文")
        return

    print(f"\n===== 找到 {len(search_results)} 篇论文 =====")
    for idx, p in enumerate(search_results[:SELECT_TOP_K], 1):
        print(f"{idx}. 标题：{p['title']}")
        print(f"   作者：{p['authors'][:3]}")
        print(f"   年份：{p['published']}")
        print(f"   URL：{p['url']}")
        print(f"   摘要：{p['summary'][:180]}...")

    actual_top_k = min(SELECT_TOP_K, len(search_results))
    confirm = input(f"\n是否下载前 {actual_top_k} 篇论文并构建知识库？（Y/N）：").strip().upper()
    if confirm != "Y":
        print("已取消")
        return

    print("\n正在下载...")
    top_papers = search_results[:actual_top_k]
    paper_metadata_list = []
    for paper in top_papers:
        metadata = PaperMetadata(
            paper_id=paper["paper_id"],
            title=paper["title"],
            authors=paper["authors"],
            pdf_url=paper["pdf_url"],
            published=str(paper["published"]) if paper["published"] is not None else "",
            primary_category=paper["primary_category"],
            summary=paper["summary"],
            published_date=paper["published_date"],
            url=paper["url"],
            categories=paper["categories"],
            doi=paper.get("doi", "")
        )
        paper_metadata_list.append(metadata)

    down_ret = await download_papers(
        papers=paper_metadata_list,
        target_dir=str(PAPERS_ROOT / GLOBAL_SESSION_ID),
        organize_by_category=False,
        filename_format="author_year_title"
    )

    if not down_ret["downloaded_files"]:
        print("没有成功下载任何论文")
        return

    print("\n正在构建知识库...")
    if await kb.build_index(down_ret["downloaded_files"]):
        print("知识库构建完成")
    else:
        print("知识库构建失败")
        return

    await query_mode(kb, memory, GLOBAL_SESSION_ID, messages)


async def query_mode(kb, memory, session_id, messages):
    """Interactive Q&A mode."""
    while True:
        print("\n进入交互问答模式（输入 quit 退出）")
        question = input("请输入您的问题：")
        if question.lower() in ["quit", "exit"]:
            memory.save(session_id, messages)
            print("\n已保存并退出")
            break
        if not question.strip():
            continue

        result = kb.query(question)

        print(f"\n【回答】\n{result['answer']}\n")
        print(f"{'='*60}")
        print(f"来源数量：{result['source_count']}")
        print(f"时间：{result['timestamp']}")
        print(f"{'='*60}")

        messages.extend([HumanMessage(question), AIMessage(result['answer'])])


if __name__ == "__main__":
    asyncio.run(main())
