"""
FastAPI Async Backend for Paper Knowledge Base Q&A System
"""
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import json

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

PROJECT_ROOT = Path(__file__).parent.parent
PAPERS_ROOT = PROJECT_ROOT / "papers"
VECTOR_DB_ROOT = PROJECT_ROOT / "vector_db"
MEMORY_ROOT = PROJECT_ROOT / "memory"

load_dotenv(PROJECT_ROOT / ".env")
sys.path.append(str(PROJECT_ROOT))

from src.tools.download_tool import download_papers, PaperMetadata
from src.tools.parse_pdf_tool import parse_pdf
from src.tools.search_tool import arxiv_search_papers

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
        List all session IDs from all sources.
        
        Returns:
            List of session IDs
        """
        sessions = set()
        
        data = self._load()
        for s in data.keys():
            sessions.add(s)
        
        if PAPERS_ROOT.exists():
            for d in PAPERS_ROOT.iterdir():
                if d.is_dir():
                    sessions.add(d.name)
        
        if VECTOR_DB_ROOT.exists():
            for d in VECTOR_DB_ROOT.iterdir():
                if d.is_dir():
                    sessions.add(d.name)
        
        return sorted(list(sessions))
    
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
            "source_count": len(docs),
            "timestamp": timestamp
        }


class SessionStore:
    """Session storage for managing multiple knowledge bases and memories."""

    def __init__(self):
        """Initialize session store."""
        self.knowledge_bases: Dict[str, PaperKnowledgeBase] = {}
        self.memories: Dict[str, List[BaseMessage]] = {}
        self.memory_manager = MemoryManager()
        self.session_models: Dict[str, str] = {}
        self.model_config_file = MEMORY_ROOT / "session_models.json"
        self._load_session_models()
    
    def _load_session_models(self):
        """Load session model configurations from file."""
        if self.model_config_file.exists():
            with open(self.model_config_file, "r", encoding="utf-8") as f:
                self.session_models = json.load(f)
    
    def _save_session_models(self):
        """Save session model configurations to file."""
        with open(self.model_config_file, "w", encoding="utf-8") as f:
            json.dump(self.session_models, f, ensure_ascii=False, indent=2)
    
    def get_session_model(self, session_id: str) -> str:
        """
        Get model name for a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Model name (defaults to MODEL_NAME from env)
        """
        if session_id in self.session_models:
            return self.session_models[session_id]
        return os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    
    def set_session_model(self, session_id: str, model_name: str) -> None:
        """
        Set model name for a session.
        
        Args:
            session_id: Session identifier
            model_name: Model name to set
        """
        self.session_models[session_id] = model_name
        self._save_session_models()

    def get_or_create_knowledge_base(self, session_id: str) -> PaperKnowledgeBase:
        """
        Get or create a knowledge base for a session.

        Args:
            session_id: Session identifier

        Returns:
            PaperKnowledgeBase instance
        """
        if session_id not in self.knowledge_bases:
            self.knowledge_bases[session_id] = PaperKnowledgeBase(session_id)
        return self.knowledge_bases[session_id]

    def get_or_create_messages(self, session_id: str) -> List[BaseMessage]:
        """
        Get or create message history for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of messages
        """
        if session_id not in self.memories:
            self.memories[session_id] = self.memory_manager.load(session_id)
        return self.memories[session_id]

    def save_messages(self, session_id: str) -> None:
        """
        Save message history for a session.

        Args:
            session_id: Session identifier
        """
        if session_id in self.memories:
            self.memory_manager.save(session_id, self.memories[session_id])
    
    def list_all_sessions(self) -> List[str]:
        """
        List all session IDs from memory storage.
        
        Returns:
            List of session IDs
        """
        return self.memory_manager.list_all_sessions()
    
    def delete_session(self, session_id: str) -> None:
        """
        Delete a session completely.
        
        Args:
            session_id: Session identifier to delete
        """
        self.memory_manager.delete_session(session_id)
    
    def rename_session(self, old_session_id: str, new_session_id: str) -> bool:
        """
        Rename a session completely.
        
        Args:
            old_session_id: Old session identifier
            new_session_id: New session identifier
            
        Returns:
            True if rename successful
        """
        if old_session_id not in self.list_all_sessions():
            return False
        
        if new_session_id in self.list_all_sessions():
            return False
        
        success = self.memory_manager.rename_session(old_session_id, new_session_id)
        
        if success:
            temp_kb = PaperKnowledgeBase(old_session_id)
            temp_kb.rename_folders(old_session_id, new_session_id)
            
            if old_session_id in self.memories:
                self.memories[new_session_id] = self.memories[old_session_id]
                del self.memories[old_session_id]
            
            if old_session_id in self.knowledge_bases:
                self.knowledge_bases[new_session_id] = PaperKnowledgeBase(new_session_id)
                del self.knowledge_bases[old_session_id]
        
        return success


app = FastAPI(
    title="论文知识库问答系统 API",
    description="基于 arXiv 论文的 RAG 问答系统，支持多会话隔离",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_store = SessionStore()


class SearchPapersRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    querys: List[str] = Field(..., description="List of search keywords")
    max_results: int = Field(default=50, description="Maximum number of results")
    sort_by: str = Field(default="Relevance", description="Sort method")
    sort_order: str = Field(default="Descending", description="Sort order")
    start_date: Optional[str] = Field(default=None, description="Start date filter")
    end_date: Optional[str] = Field(default=None, description="End date filter")


class DownloadPapersRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    paper_indices: List[int] = Field(..., description="List of paper indices to download (0-based)")
    search_results: List[Dict[str, Any]] = Field(..., description="Search results list")


class BuildKnowledgeBaseRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    clear_existing: bool = Field(default=False, description="Whether to clear existing knowledge base")


class QueryRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    question: str = Field(..., description="User question")


class SetModelRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    model_name: str = Field(..., description="Model name to use")


class SessionInfoRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier")


class CreateSessionRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier")


class RenameSessionRequest(BaseModel):
    old_session_id: str = Field(..., description="Old session identifier")
    new_session_id: str = Field(..., description="New session identifier")


@app.get("/")
async def root():
    """根端点，返回 API 信息。"""
    return {
        "message": "论文知识库问答系统 API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/api/sessions")
async def list_sessions():
    """
    Get list of all session IDs.
    
    Returns:
        List of session IDs
    """
    try:
        sessions = session_store.list_all_sessions()
        return {
            "success": True,
            "data": {
                "sessions": sessions,
                "count": len(sessions)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/create")
async def create_session(request: CreateSessionRequest):
    """
    Create a new session and save it to memory.
    
    Args:
        request: CreateSessionRequest containing session_id
    
    Returns:
        Success message
    """
    try:
        kb = session_store.get_or_create_knowledge_base(request.session_id)
        messages = session_store.get_or_create_messages(request.session_id)
        session_store.save_messages(request.session_id)
        
        return {
            "success": True,
            "message": "会话创建成功",
            "data": {
                "session_id": request.session_id
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/info")
async def get_session_info(request: SessionInfoRequest):
    """Get session information."""
    try:
        kb = session_store.get_or_create_knowledge_base(request.session_id)
        messages = session_store.get_or_create_messages(request.session_id)
        model_name = session_store.get_session_model(request.session_id)

        return {
            "success": True,
            "data": {
                "session_id": request.session_id,
                "has_knowledge": kb.has_exist_knowledge(),
                "papers": kb.list_exist_papers(),
                "message_count": len(messages),
                "model_name": model_name
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/model")
async def get_session_model_endpoint(request: SessionInfoRequest):
    """Get model for a session."""
    try:
        model_name = session_store.get_session_model(request.session_id)
        return {
            "success": True,
            "data": {
                "session_id": request.session_id,
                "model_name": model_name
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/set_model")
async def set_session_model(request: SetModelRequest):
    """Set model for a session."""
    try:
        session_store.set_session_model(request.session_id, request.model_name)
        return {
            "success": True,
            "message": "模型设置成功",
            "data": {
                "session_id": request.session_id,
                "model_name": request.model_name
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/clear")
async def clear_session(request: SessionInfoRequest):
    """清除会话资源。"""
    try:
        kb = session_store.get_or_create_knowledge_base(request.session_id)
        kb.clear_old_knowledge()

        if request.session_id in session_store.memories:
            del session_store.memories[request.session_id]
        if request.session_id in session_store.knowledge_bases:
            del session_store.knowledge_bases[request.session_id]
        
        session_store.delete_session(request.session_id)

        return {
            "success": True,
            "message": "会话资源已清除"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/rename")
async def rename_session(request: RenameSessionRequest):
    try:
        success = session_store.rename_session(request.old_session_id, request.new_session_id)
        
        if success:
            return {
                "success": True,
                "message": "会话重命名成功",
                "data": {
                    "old_session_id": request.old_session_id,
                    "new_session_id": request.new_session_id
                }
            }
        else:
            return {
                "success": False,
                "message": "会话重命名失败，请检查会话ID是否正确"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/papers/search")
async def search_papers(request: SearchPapersRequest):
    """Search arXiv papers."""
    try:
        search_results = await arxiv_search_papers(
            querys=request.querys,
            max_results=request.max_results,
            sort_by=request.sort_by,
            sort_order=request.sort_order,
            start_date=request.start_date,
            end_date=request.end_date
        )

        return {
            "success": True,
            "data": {
                "count": len(search_results),
                "papers": search_results
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/papers/download")
async def download_selected_papers(request: DownloadPapersRequest):
    """Download selected papers."""
    try:
        kb = session_store.get_or_create_knowledge_base(request.session_id)

        target_dir = str(PAPERS_ROOT / request.session_id)

        selected_papers = [request.search_results[i] for i in request.paper_indices]

        paper_metadata_list = []
        for paper in selected_papers:
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
            target_dir=target_dir,
            organize_by_category=False,
            filename_format="author_year_title"
        )

        return {
            "success": True,
            "data": {
                "success": down_ret["success"],
                "failed": down_ret["failed"],
                "skipped": down_ret["skipped"],
                "downloaded_files": down_ret["downloaded_files"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/knowledge/build")
async def build_knowledge_base(request: BuildKnowledgeBaseRequest):
    """Build knowledge base."""
    try:
        kb = session_store.get_or_create_knowledge_base(request.session_id)

        if request.clear_existing:
            kb.clear_old_knowledge()
            await asyncio.sleep(0.5)

        kb.papers_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = []
        max_attempts = 3
        for attempt in range(max_attempts):
            pdf_files = [str(f) for f in kb.papers_dir.glob("*.pdf")]
            if pdf_files:
                break
            await asyncio.sleep(0.5)

        if not pdf_files:
            all_files = list(kb.papers_dir.glob("*"))
            return {
                "success": False,
                "message": f"未找到 PDF 文件。目录：{str(kb.papers_dir)}，是否存在：{kb.papers_dir.exists()}"
            }

        result = await kb.build_index(pdf_files)

        return {
            "success": result,
            "data": {
                "pdf_count": len(pdf_files),
                "pdf_files": pdf_files
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/knowledge/load")
async def load_knowledge_base(request: SessionInfoRequest):
    """加载知识库。"""
    try:
        kb = session_store.get_or_create_knowledge_base(request.session_id)

        if not kb.has_exist_knowledge():
            return {
                "success": False,
                "message": "未找到知识库"
            }

        kb.load_index()

        return {
            "success": True,
            "message": "知识库加载成功"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def query_knowledge_base(request: QueryRequest):
    """查询知识库。"""
    try:
        kb = session_store.get_or_create_knowledge_base(request.session_id)
        messages = session_store.get_or_create_messages(request.session_id)
        model_name = session_store.get_session_model(request.session_id)
        chat_model = create_chat_model(model_name)

        if not kb.vector_store:
            if kb.has_exist_knowledge():
                kb.load_index()
            else:
                return {
                    "success": False,
                    "message": "知识库未加载，请先构建或加载"
                }

        retriever = kb.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

        docs = retriever.invoke(request.question)
        
        papers_info = kb._get_papers_info()

        prompt = ChatPromptTemplate.from_template("""
你是一个知识渊博的AI助手。请仅根据提供的上下文用中文回答问题。

上下文: {context}

当前知识库包含的论文:
{papers_info}

问题: {question}

说明:
1. 仅根据上下文回答，不要编造或使用外部信息
2. 如果上下文中没有相关信息，请这样回答：
   "抱歉，当前知识库中没有找到与您问题相关的内容。当前知识库包含以下论文：
   {papers_info}
   如果您想了解其他主题，请先搜索并下载相关论文来更新知识库。"
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
            | chat_model
            | StrOutputParser()
        )

        answer = rag_chain.invoke({"question": request.question})

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
        
        result = {
            "question": request.question,
            "answer": answer,
            "source_count": len(docs),
            "timestamp": timestamp
        }

        messages.extend([HumanMessage(request.question), AIMessage(answer)])
        session_store.save_messages(request.session_id)

        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/messages/{session_id}")
async def get_messages(session_id: str):
    try:
        messages = session_store.get_or_create_messages(session_id)

        return {
            "success": True,
            "data": {
                "messages": [
                    {"role": m.type, "content": m.content}
                    for m in messages
                ]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
