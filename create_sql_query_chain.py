from langchain_core.prompts import ChatPromptTemplate
import re

# ---------- 只读 SQL 校验规则 ----------
READ_ONLY_PATTERN = re.compile(r"^\s*(select|with)\s", re.IGNORECASE)

FORBIDDEN_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter",
    "truncate", "replace", "create", "attach", "detach",
    "pragma", "vacuum"
]


def _sanitize_and_validate_sql(sql: str) -> str:
    """清洗并校验 SQL，只允许只读操作"""
    if not sql:
        raise ValueError("SQL 为空")

    # 去掉 markdown 包裹
    sql = sql.strip()
    if sql.startswith("```"):
        sql = sql.strip("`")
        sql = sql.replace("sql", "", 1).strip()

    # 只保留第一条语句
    sql = sql.split(";")[0].strip() + ";"

    sql_lower = sql.lower()

    # 必须以 SELECT / WITH 开头
    if not READ_ONLY_PATTERN.match(sql_lower):
        raise ValueError("检测到非只读 SQL（仅允许 SELECT / WITH）")

    # 禁止危险关键字
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", sql_lower):
            raise ValueError(f"检测到危险 SQL 关键字: {kw}")

    return sql


def create_sql_query_chain(llm, db):
    """
    安全 SQL 生成器
    输入: {"question": str}
    输出: str (安全、只读 SQL)
    """

    prompt = ChatPromptTemplate.from_template("""
你是一个数据库专家，只负责生成 SQL 查询语句。

数据库类型：{dialect}

【数据库表结构】
{table_info}

【用户问题】
{question}

【严格要求】
- 只生成 SELECT / WITH 查询
- 严禁修改数据库
- 不要使用 markdown
- 不要解释
- 只输出一条 SQL
""".strip())

    class SQLQueryChain:
        def invoke(self, inputs: dict) -> str:
            question = inputs.get("question")
            if not question:
                raise ValueError("缺少 question")

            messages = prompt.format_messages(
                dialect=db.dialect,
                table_info=db.get_table_info(),
                question=question
            )

            response = llm.invoke(messages)

            raw_sql = response.content
            safe_sql = _sanitize_and_validate_sql(raw_sql)

            return safe_sql

        def __call__(self, inputs: dict) -> str:
            return self.invoke(inputs)

    return SQLQueryChain()

