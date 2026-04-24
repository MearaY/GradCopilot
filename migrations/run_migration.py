"""
数据库迁移脚本：执行 migrations/001_init_schema.sql。

使用方式：
    python migrations/run_migration.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from sqlalchemy import create_engine, text

DATABASE_URL = os.environ["POSTGRES_URL"]
engine = create_engine(DATABASE_URL)

SQL_FILE = Path(__file__).parent / "001_init_schema.sql"


def run():
    print(f"[Migration] 连接数据库：{DATABASE_URL}")
    sql = SQL_FILE.read_text(encoding="utf-8")

    # 按分号分割，逐条执行（跳过注释行）
    statements = []
    current = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        current.append(line)
        if stripped.endswith(";"):
            statements.append("\n".join(current))
            current = []

    with engine.begin() as conn:
        for i, stmt in enumerate(statements, 1):
            try:
                conn.execute(text(stmt))
                print(f"  [{i}/{len(statements)}] OK")
            except Exception as e:
                print(f"  [{i}/{len(statements)}] WARN: {e}")

    print("[Migration] 完成。")


if __name__ == "__main__":
    run()
