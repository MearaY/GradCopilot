"""
PostgreSQL 连接管理。

使用 SQLAlchemy + pgvector，连接池配置：pool_size=5, max_overflow=10。
POSTGRES_URL 必须配置，未配置时启动即报错。
"""
import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

# 必填环境变量，未配置时启动即报 KeyError
DATABASE_URL: str = os.environ["POSTGRES_URL"]

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,  # 每次使用前检测连接存活
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """
    FastAPI Depends 用法：
        db: Session = Depends(get_db)

    Refs:
        使用方式：db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_connection() -> bool:
    """
    验证 PostgreSQL 连接是否正常，用于 /api/health。

    Returns:
        bool: True 表示连接正常

    Refs:
        用于 /api/health 健康检查接口。
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
