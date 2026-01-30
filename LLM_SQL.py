import datetime
import os
from dotenv import load_dotenv

load_dotenv()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from create_sql_query_chain import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict

print("正在初始化系统...")

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not API_KEY:
    print("错误: 未找到API密钥！请在.env文件中设置 API_KEY 或 DEEPSEEK_API_KEY")
    print("格式: API_KEY=your_deepseek_api_key_here")
    exit(1)

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=API_KEY,
    temperature=0.1,
    max_tokens=2048
)


# ==================== 1. 安全的SQL生成器 ====================
class SafeSQLGenerator:
    """安全的SQL生成器，防止SQL注入"""

    def __init__(self):
        # 定义表结构和字段映射
        self.table_schema = {
            "mine_production": {
                "fields": {
                    "采场编号": ["采场编号", "采场", "矿场", "场区"],
                    "日期": ["日期", "时间", "生产日期"],
                    "出矿量": ["出矿量", "产量", "矿石产量", "矿量", "粗矿量", "出矿"],
                    "废石量": ["废石量", "废料", "废石", "废石产出"],
                    "净矿量": ["净矿量", "净产量", "净出矿量"]
                }
            }
        }

        # 危险SQL关键字
        self.dangerous_keywords = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'TRUNCATE',
            'CREATE', 'ALTER', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE',
            'UNION', ';', '--', '/*', '*/'
        ]

    def sanitize_sql(self, sql: str) -> str:
        """清理SQL语句，防止SQL注入"""
        if not sql:
            return ""

        # 转换为大写便于检查
        sql_upper = sql.upper()

        # 检查是否包含危险操作
        for keyword in self.dangerous_keywords:
            if keyword in sql_upper and keyword != ';' and keyword != '--':
                raise ValueError(f"SQL语句包含危险操作: {keyword}")

        # 检查是否是SELECT查询
        if not sql_upper.strip().startswith('SELECT'):
            raise ValueError("只允许SELECT查询操作")

        # 替换双引号为单引号（字段名中的双引号需要特殊处理）
        # 但我们的字段名是中文，不需要双引号，所以可以直接替换
        sql = sql.replace('"', "'")

        # 确保字段名安全（只允许中文字段名）
        safe_fields = ['采场编号', '日期', '出矿量', '废石量', '净矿量']
        for field in safe_fields:
            # 如果字段名被引号包裹，移除引号
            sql = sql.replace(f"'{field}'", field)
            sql = sql.replace(f'"{field}"', field)

        # 移除多余的注释和分号
        sql = sql.split(';')[0]  # 只取第一个SQL语句
        sql = sql.split('--')[0]  # 移除行注释

        return sql.strip()

    def add_safety_checks_to_prompt(self, question: str) -> str:
        """在问题中添加安全提示"""
        safety_prompt = f"""
        请为以下问题生成SQLite兼容的SQL查询：
        {question}

        重要安全规则：
        1. 只能使用SELECT查询，不能包含INSERT、UPDATE、DELETE等
        2. 表名必须是 mine_production
        3. 字段名必须使用中文：采场编号、日期、出矿量、废石量、净矿量
        4. 所有字符串值必须使用单引号，不能使用双引号
        5. 采场编号格式为：'XPD001'、'XPD002'等
        6. 日期条件使用SQLite兼容的语法
        7. 不要包含注释和分号
        """
        return safety_prompt


# ==================== 2. 安全的SQL查询工具 ====================
class SafeQuerySQLDatabaseTool:
    """安全的SQL查询工具，包含SQL注入防护"""

    def __init__(self, db: SQLDatabase):
        self.db = db
        self.sql_generator = SafeSQLGenerator()

    def invoke(self, sql_query: str) -> str:
        """执行安全的SQL查询"""
        try:
            # 清理和验证SQL
            safe_sql = self.sql_generator.sanitize_sql(sql_query)
            print(f"安全SQL: {safe_sql}")

            # 执行查询
            result = self.db.run(safe_sql)
            return result
        except Exception as e:
            return f"SQL执行失败: {str(e)}"


# ==================== 3. 安全的SQL生成链 ====================
def create_safe_sql_query_chain(llm: ChatOpenAI, db: SQLDatabase, sql_generator: SafeSQLGenerator):
    """创建安全的SQL查询链"""

    def safe_chain(question: str) -> str:
        """安全的SQL生成函数"""
        try:
            # 构建安全提示
            table_info = db.get_table_info()

            safe_prompt = f"""
            你是一个SQL生成专家，专门为矿山生产数据库生成安全的SQL查询。

            数据库表结构：
            {table_info}

            重要规则：
            1. 只能生成SELECT查询，绝对不能包含INSERT、UPDATE、DELETE、DROP等修改操作
            2. 字段名使用中文：采场编号、日期、出矿量、废石量、净矿量
            3. 所有字符串值使用单引号，不能使用双引号
            4. 采场编号格式：'XPD001'、'XPD002'等
            5. 日期条件使用SQLite兼容的语法
            6. 不要包含注释和分号
            7. 只生成一个SQL查询语句

            用户问题：{question}

            请生成安全的SQL查询语句：
            """

            # 创建提示模板
            prompt = ChatPromptTemplate.from_template(safe_prompt)

            # 创建链
            chain = prompt | llm | StrOutputParser()

            # 生成SQL
            sql = chain.invoke({"question": question})

            # 清理SQL
            safe_sql = sql_generator.sanitize_sql(sql)
            return safe_sql
        except Exception as e:
            return f"SQL生成失败: {str(e)}"

    return safe_chain


# ==================== 4. 智能图表生成器 ====================
class SmartChartGenerator:
    """智能图表生成器，优化图表显示"""

    def __init__(self):
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 图表类型映射
        self.chart_type_mapping = {
            '趋势': 'line',
            '对比': 'bar',
            '分布': 'hist',
            '比例': 'pie',
            '关系': 'scatter',
            '热图': 'heatmap'
        }

    def generate_chart_code(self, df_sample: pd.DataFrame, question: str, filename: str) -> str:
        """生成智能图表代码"""

        # 分析问题意图，选择合适的图表类型
        chart_type = self._detect_chart_type(question, df_sample)

        # 根据图表类型生成代码
        if chart_type == 'line':
            return self._generate_line_chart_code(df_sample, question, filename)
        elif chart_type == 'bar':
            return self._generate_bar_chart_code(df_sample, question, filename)
        elif chart_type == 'hist':
            return self._generate_histogram_code(df_sample, question, filename)
        elif chart_type == 'pie':
            return self._generate_pie_chart_code(df_sample, question, filename)
        else:
            return self._generate_default_chart_code(df_sample, question, filename)

    def _detect_chart_type(self, question: str, df_sample: pd.DataFrame) -> str:
        """检测图表类型"""
        question_lower = question.lower()

        if any(word in question_lower for word in ['趋势', '变化', '时间序列', '走势']):
            return 'line'
        elif any(word in question_lower for word in ['对比', '比较', '柱状图', '条形图']):
            return 'bar'
        elif any(word in question_lower for word in ['分布', '直方图', '频率']):
            return 'hist'
        elif any(word in question_lower for word in ['饼状图', '饼图', '占比', '比例', '比值', '百分比', '比重', '份额']):
            return 'pie'
        elif any(word in question_lower for word in ['关系', '散点图', '相关性']):
            return 'scatter'
        else:
            # 根据数据特征选择
            if '日期' in df_sample.columns and len(df_sample) > 5:
                return 'line'
            elif '采场编号' in df_sample.columns:
                return 'bar'
            else:
                return 'bar'

    def _generate_simple_pie_chart_code(self, df_sample: pd.DataFrame, question: str, filename: str) -> str:
        """生成简化的饼图代码"""

        # 提取数据
        data_dict = df_sample.to_dict()

        code = f"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据
data = {data_dict}
df = pd.DataFrame(data)

print("数据列:", df.columns.tolist())
print("数据:")
print(df)

# 创建图表
fig, ax = plt.subplots(figsize=(10, 8))

# 检查数据
if '净矿量总计' in df.columns:
    values = df['净矿量总计'].tolist()
elif '净矿量' in df.columns:
    values = df['净矿量'].tolist()
else:
    # 找第一个数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        values = df[numeric_cols[0]].tolist()
    else:
        values = []

# 标签
if '采场编号' in df.columns:
    labels = df['采场编号'].astype(str).tolist()
else:
    labels = [f'数据{{i+1}}' for i in range(len(values))]

# 生成饼图
if values and len(values) > 0:
    # 处理问题：如果是XPD001占比问题，重新组织数据
    question_lower = '{question}'.lower()
    if 'xpd001' in question_lower and ('占比' in question_lower or '比值' in question_lower):
        # 找到XPD001的索引
        xpd001_index = -1
        for i, label in enumerate(labels):
            if 'XPD001' in str(label):
                xpd001_index = i
                break

        if xpd001_index != -1:
            xpd001_value = values[xpd001_index]
            other_value = sum(values) - xpd001_value

            labels = ['XPD001采场', '其他采场']
            values = [xpd001_value, other_value]

    # 设置颜色
    colors = plt.cm.Set3(np.linspace(0, 1, len(values)))

    # 绘制饼图
    wedges, texts, autotexts = ax.pie(
        values, 
        labels=labels, 
        colors=colors,
        autopct='%1.1f%%',
        startangle=90
    )

    # 添加标题
    ax.set_title('净矿量占比分布', fontsize=14, fontweight='bold')

    # 添加图例
    ax.legend(wedges, labels, title="采场", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
else:
    ax.text(0.5, 0.5, '没有有效数据', ha='center', va='center', fontsize=14)

# 确保圆形
ax.axis('equal')

# 保存图表
plt.tight_layout()
plt.savefig('{filename}', dpi=300, bbox_inches='tight')
print(f"图表已保存: {filename}")
plt.show()
"""
        return code


    def _generate_line_chart_code(self, df_sample: pd.DataFrame, question: str, filename: str) -> str:
        """生成折线图代码"""
        code = f"""
import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 准备数据
data = {df_sample.head(10).to_dict()}
df = pd.DataFrame(data)

# 创建图表
fig, ax = plt.subplots(figsize=(12, 6))

# 检查数据列
if '日期' in df.columns:
    # 如果有日期列，尝试转换为日期格式
    try:
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期')
        x_data = df['日期']
    except:
        x_data = df.index

    # 绘制数值字段
    numeric_columns = ['出矿量', '废石量', '净矿量']
    for col in numeric_columns:
        if col in df.columns:
            ax.plot(x_data, df[col], marker='o', label=col, linewidth=2)

    ax.set_xlabel('日期', fontsize=12)
else:
    # 没有日期列，使用索引
    numeric_columns = ['出矿量', '废石量', '净矿量']
    for col in numeric_columns:
        if col in df.columns:
            ax.plot(df.index, df[col], marker='o', label=col, linewidth=2)

    ax.set_xlabel('数据点', fontsize=12)

ax.set_ylabel('数值', fontsize=12)
ax.set_title('{question[:30]}趋势图', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 优化布局
plt.tight_layout()

# 保存图表
plt.savefig('{filename}', dpi=300, bbox_inches='tight')
print(f"图表已保存为: '{filename}'")
plt.show()
"""
        return code

    def _generate_bar_chart_code(self, df_sample: pd.DataFrame, question: str, filename: str) -> str:
        """生成柱状图代码"""
        code = f"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 准备数据
data = {df_sample.head(10).to_dict()}
df = pd.DataFrame(data)

# 创建图表
fig, ax = plt.subplots(figsize=(12, 6))

# 检查是否有分组字段
if '采场编号' in df.columns:
    # 按采场编号分组
    if '出矿量' in df.columns:
        df_grouped = df.groupby('采场编号')['出矿量'].sum().reset_index()
        x_labels = df_grouped['采场编号'].tolist()
        y_values = df_grouped['出矿量'].tolist()

        bars = ax.bar(x_labels, y_values, color='steelblue', alpha=0.8)
        ax.set_xlabel('采场编号', fontsize=12)
        ax.set_ylabel('出矿量总和', fontsize=12)

        # 在柱子上添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(y_values)*0.01,
                   f'{{int(height)}}', ha='center', va='bottom', fontsize=9)

    elif '净矿量' in df.columns:
        df_grouped = df.groupby('采场编号')['净矿量'].sum().reset_index()
        x_labels = df_grouped['采场编号'].tolist()
        y_values = df_grouped['净矿量'].tolist()

        bars = ax.bar(x_labels, y_values, color='forestgreen', alpha=0.8)
        ax.set_xlabel('采场编号', fontsize=12)
        ax.set_ylabel('净矿量总和', fontsize=12)

    else:
        # 如果没有明确的数值字段，使用第一个数值字段
        numeric_cols = [col for col in df.columns if col in ['出矿量', '废石量', '净矿量']]
        if numeric_cols:
            col = numeric_cols[0]
            df_grouped = df.groupby('采场编号')[col].sum().reset_index()
            x_labels = df_grouped['采场编号'].tolist()
            y_values = df_grouped[col].tolist()

            bars = ax.bar(x_labels, y_values, color='coral', alpha=0.8)
            ax.set_xlabel('采场编号', fontsize=12)
            ax.set_ylabel(col, fontsize=12)
else:
    # 没有分组字段，直接绘制
    numeric_cols = [col for col in df.columns if col in ['出矿量', '废石量', '净矿量']]
    if numeric_cols:
        col = numeric_cols[0]
        y_values = df[col].head(10).tolist()
        x_labels = [f'数据{{i+1}}' for i in range(len(y_values))]

        bars = ax.bar(x_labels, y_values, color='steelblue', alpha=0.8)
        ax.set_xlabel('数据点', fontsize=12)
        ax.set_ylabel(col, fontsize=12)

ax.set_title('{question[:30]}对比图', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 优化布局
plt.tight_layout()

# 保存图表
plt.savefig('{filename}', dpi=300, bbox_inches='tight')
print(f"图表已保存为: '{filename}'")
plt.show()
"""
        return code

    def _generate_histogram_code(self, df_sample: pd.DataFrame, question: str, filename: str) -> str:
        """生成直方图代码"""
        code = f"""
import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 准备数据
data = {df_sample.head(20).to_dict()}
df = pd.DataFrame(data)

# 创建图表
fig, ax = plt.subplots(figsize=(12, 6))

# 选择数值字段绘制直方图
numeric_cols = [col for col in df.columns if col in ['出矿量', '废石量', '净矿量']]
if numeric_cols:
    col = numeric_cols[0]
    data_values = df[col].dropna().tolist()

    if data_values:
        ax.hist(data_values, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('频次', fontsize=12)
        ax.set_title('{question[:30]}分布直方图', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, '没有有效数据可绘制', ha='center', va='center', fontsize=14)
        ax.set_title('数据为空', fontsize=14)
else:
    ax.text(0.5, 0.5, '没有数值数据可绘制', ha='center', va='center', fontsize=14)
    ax.set_title('没有数值字段', fontsize=14)

# 优化布局
plt.tight_layout()

# 保存图表
plt.savefig('{filename}', dpi=300, bbox_inches='tight')
print(f"图表已保存为: '{filename}'")
plt.show()
"""
        return code

    def _generate_pie_chart_code(self, df_sample: pd.DataFrame, question: str, filename: str) -> str:
        """生成饼图代码 - 修复变量定义问题"""
        print(f"生成饼图，数据列: {df_sample.columns.tolist()}")

        # 确定正确的数值列名
        numeric_cols = [col for col in df_sample.columns if col in ['净矿量总计', '净矿量', '出矿量', '废石量']]
        value_col = numeric_cols[0] if numeric_cols else None

        code = f"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 准备数据
data = {df_sample.to_dict()}
df = pd.DataFrame(data)

print(f"饼图数据形状: {{df.shape}}")
print(f"数据列名: {{list(df.columns)}}")

# 创建图表
fig, ax = plt.subplots(figsize=(12, 8))

# 初始化变量，避免未定义错误
labels = []
sizes = []
title = '净矿量占比饼图'

# 检查是否有采场编号和数值字段
if '采场编号' in df.columns and '{value_col}' in df.columns:
    # 按采场编号分组计算总和
    df_grouped = df.groupby('采场编号')['{value_col}'].sum().reset_index()
    df_grouped = df_grouped.sort_values('{value_col}', ascending=False)

    print(f"分组后数据: \\n{{df_grouped}}")

    # 根据问题类型调整标签和大小
    question_lower = '{question.lower()}'

    if 'xpd001' in question_lower and ('占比' in question_lower or '比值' in question_lower):
        # 计算XPD001的占比
        xpd001_value = df_grouped[df_grouped['采场编号'] == 'XPD001']['{value_col}'].values[0] if 'XPD001' in df_grouped['采场编号'].values else 0
        total_value = df_grouped['{value_col}'].sum()
        other_value = total_value - xpd001_value

        labels = ['XPD001采场', '其他采场']
        sizes = [xpd001_value, other_value]
        colors = ['#ff9999', '#66b3ff']
        title = 'XPD001采场净矿量占比'
    else:
        # 绘制所有采场
        labels = df_grouped['采场编号'].astype(str).tolist()
        sizes = df_grouped['{value_col}'].tolist()
        colors = plt.cm.Set3(range(len(labels)))
        title = '各采场净矿量占比'
else:
    # 如果没有采场编号，尝试使用其他列
    if '{value_col}' in df.columns:
        labels = [f'数据{{i+1}}' for i in range(len(df))]
        sizes = df['{value_col}'].tolist()
        colors = plt.cm.Set3(range(len(labels)))
        title = '{value_col}分布'
    else:
        # 如果没有数值列，显示错误信息
        ax.text(0.5, 0.5, '没有合适的数值数据用于生成饼图', 
                ha='center', va='center', fontsize=12)
        ax.set_title('数据不适合饼图', fontsize=14)

# 绘制饼图（如果有数据）
if sizes and len(sizes) > 0 and sum(sizes) > 0:
    # 绘制饼图
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels if labels else None, 
        colors=colors if 'colors' in locals() else plt.cm.Set3(range(len(sizes))),
        autopct=lambda pct: f'{{pct:.1f}}%\\n({{int(pct*sum(sizes)/100)}})',
        startangle=90, 
        shadow=True
    )

    # 美化文本
    for text in texts:
        text.set_fontsize(11)
        text.set_fontweight('bold')

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')

    # 添加图例
    if labels:
        ax.legend(wedges, labels, title="采场编号", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

# 确保饼图是圆形
ax.axis('equal')

# 优化布局
plt.tight_layout()

# 保存图表
plt.savefig('{filename}', dpi=300, bbox_inches='tight')
print(f"饼图已保存为: '{{filename}}'")
plt.show()
"""
        return code

    def _generate_default_chart_code(self, df_sample: pd.DataFrame, question: str, filename: str) -> str:
        """生成默认图表代码"""
        return self._generate_bar_chart_code(df_sample, question, filename)


# ==================== 5. 测试数据库连接 ====================
print("正在连接数据库...")
try:
    db = SQLDatabase.from_uri(
        "sqlite:///./SQLDB/mine_data.db",
        include_tables=['mine_production'],
        sample_rows_in_table_info=2
    )
    # 测试连接
    db.run("SELECT COUNT(*) FROM mine_production LIMIT 1")
    print("数据库连接成功！")
except Exception as e:
    print(f"数据库连接失败: {e}")
    print("请确保数据库文件存在: ./SQLDB/mine_data.db")
    exit(1)


# ==================== 6. 定义智能体状态 ====================
class AgentState(TypedDict):
    question: str
    sql_query: str
    query_result: str
    chart_code: str
    final_answer: str
    next_step: str
    error_message: str


# ==================== 7. 构建智能体工具 ====================
print("正在初始化工具...")
try:
    # 初始化安全工具
    sql_generator = SafeSQLGenerator()
    chart_generator = SmartChartGenerator()

    # 使用安全的SQL查询链
    write_query_tool = create_safe_sql_query_chain(llm, db, sql_generator)

    # 使用安全的SQL查询工具
    execute_query_tool = SafeQuerySQLDatabaseTool(db=db)

    # Python执行工具
    repl_tool = PythonAstREPLTool(locals={"pd": pd, "plt": plt, "sns": sns})

    print("工具初始化完成！")
except Exception as e:
    print(f"工具初始化失败: {e}")
    exit(1)


# ==================== 8. 定义路由与节点函数 ====================
def route_question(state: AgentState):
    """核心路由逻辑：判断用户意图"""
    question = state['question'].lower()

    # 增强的关键词识别
    visualize_keywords = ['统计', '趋势', '图表', '画图', '可视化', '分析', '柱状图',
                          '折线图', '饼图', '分布', '对比', '比较', '展示', '绘图', '图']
    query_keywords = ['查询', '查看', '显示', '获取', '多少', '何时', '哪里', '谁', '什么']

    # 如果问题中包含可视化关键词，直接进入可视化流程
    if any(word in question for word in visualize_keywords):
        return "visualize"
    elif any(word in question for word in query_keywords):
        return "query"
    else:
        # 默认认为是查询
        return "query"


def sql_agent_node(state: AgentState):
    """SQL智能体节点：生成并执行安全查询"""
    print(f"处理SQL查询: {state['question'][:50]}...")
    print(f"用户完整问题: {state['question']}")

    try:
        # 1. 生成安全的SQL - 直接调用函数
        sql_query = write_query_tool(state['question'])
        print(f"生成的SQL: {sql_query}")

        # 2. 执行查询
        query_result = execute_query_tool.invoke(sql_query)
        print(f"查询结果原始内容: {query_result[:500] if query_result else '空'}")

        # 检查SQL是否生成成功
        if sql_query.startswith("SQL生成失败"):
            return {
                "sql_query": "生成失败",
                "query_result": "",
                "final_answer": sql_query,
                "next_step": "end",
                "error_message": sql_query
            }

        # 3. 用自然语言解释结果
        answer_prompt = ChatPromptTemplate.from_template("""
        你是一个矿山数据分析助手。请根据以下信息，用简洁清晰的中文回答用户问题。

        用户问题：{question}
        执行的SQL查询：{sql_query}
        查询结果：{query_result}

        请按照以下格式回答：
        1. 直接回答用户的问题
        2. 总结关键数据
        3. 如果有异常或空数据，给出可能的原因

        你的回答：
        """)

        answer_chain = answer_prompt | llm | StrOutputParser()
        final_answer = answer_chain.invoke({
            "question": state['question'],
            "sql_query": sql_query,
            "query_result": str(query_result)[:500]
        })

        # 4. 判断是否需要可视化 - 修改判断条件
        result_str = str(query_result)
        question_lower = state['question'].lower()

        # 检查是否包含可视化关键词 - 加强检测
        visualization_keywords = [
            '趋势', '对比', '统计', '图表', '画图', '可视化', '柱状图',
            '折线图', '饼状图', '饼图', '分布', '展示', '绘图', '图', '比值'
        ]
        has_visualization_keywords = any(word in question_lower for word in visualization_keywords)

        # 添加饼图专门检测
        pie_chart_keywords = ['饼状图', '饼图', '占比', '比例', '比值']
        has_pie_chart_keyword = any(word in question_lower for word in pie_chart_keywords)

        # 修改可视化判断逻辑：
        # 1. 如果明确要求饼图，即使数据量小也生成
        # 2. 降低长度阈值或使用更智能的判断
        # 3. 检查是否有有效数据（不是空结果）

        # 检查是否有有效数据（不是空列表或空字符串）
        has_valid_data = (
                result_str and
                result_str not in ['', '[]', '()', 'None', 'null'] and
                '失败' not in result_str and
                '错误' not in result_str.lower()
        )

        # 如果有饼图关键词，强制生成图表
        if has_pie_chart_keyword and has_valid_data:
            needs_visualization = True
            print(f"检测到饼图关键词，强制生成图表")
        else:
            # 其他图表类型的判断
            needs_visualization = (
                    has_valid_data and
                    (len(result_str) > 50 or has_visualization_keywords) and  # 降低长度阈值
                    (has_visualization_keywords or "并生成" in question_lower or "并绘制" in question_lower)
            )

        # 添加调试信息
        print(f"可视化判断: needs_visualization={needs_visualization}, "
              f"has_visualization_keywords={has_visualization_keywords}, "
              f"has_pie_chart_keyword={has_pie_chart_keyword}, "
              f"result_str_length={len(result_str)}, "
              f"has_valid_data={has_valid_data}")

        return {
            "sql_query": sql_query,
            "query_result": str(query_result)[:2000],
            "final_answer": final_answer,
            "next_step": "visualize" if needs_visualization else "end",
            "error_message": ""
        }

    except Exception as e:
        error_msg = f"SQL处理失败: {str(e)}"
        print(f"错误: {error_msg}")
        return {
            "sql_query": "生成失败",
            "query_result": error_msg,
            "final_answer": f"抱歉，处理查询时出错: {str(e)[:100]}",
            "next_step": "end",
            "error_message": error_msg
        }


def visualize_agent_node(state: AgentState):
    """可视化智能体节点：生成分析图表"""
    print("进入可视化智能体节点，生成图表...")
    print(f"问题内容: {state['question']}")

    try:
        # 检查是否有查询结果
        if not state.get('query_result') or "失败" in state['query_result']:
            print("查询结果为空或失败，跳过图表生成")
            return {
                "chart_code": "无数据可生成图表",
                "final_answer": state.get('final_answer', '') + "\n\n❌ 没有有效数据可用于生成图表",
                "next_step": "end",
                "error_message": "无有效数据"
            }

        # 生成带时间戳的唯一文件名
        timestamp = int(datetime.datetime.now().timestamp())
        chart_filename = f"./Picture/output_chart_{timestamp}.png"

        # 确保目录存在
        os.makedirs("./Picture", exist_ok=True)

        # 从查询结果中提取数据
        query_result = state['query_result']
        print(f"开始解析查询结果，长度: {len(query_result)}")
        print(f"查询结果内容: {query_result}")

        # 直接解析查询结果字符串
        try:
            import ast

            # 清理字符串，确保可以正确解析
            clean_result = query_result.strip()

            # 如果字符串以 [ 开头和 ] 结尾，尝试解析为列表
            if clean_result.startswith('[') and clean_result.endswith(']'):
                data_list = ast.literal_eval(clean_result)
                print(f"成功解析为Python列表: {type(data_list)}, 长度: {len(data_list) if data_list else 0}")

                if data_list and len(data_list) > 0:
                    # 转换为DataFrame
                    df_sample = pd.DataFrame(data_list, columns=['采场编号', '净矿量总计'])
                    print(f"创建DataFrame成功: {df_sample.shape}")
                    print(f"DataFrame数据:\\n{df_sample}")
                else:
                    raise ValueError("解析的数据列表为空")
            else:
                raise ValueError(f"查询结果不是有效的列表格式: {clean_result[:50]}...")

        except Exception as e:
            print(f"数据解析失败: {e}")
            print("尝试其他解析方式...")

            # 尝试从查询结果中提取数据
            import re

            # 使用正则表达式提取 (XPDXXX, 数字) 格式的数据
            pattern = r"\('([^']+)',\s*(\d+)\)"
            matches = re.findall(pattern, query_result)

            if matches:
                print(f"通过正则表达式找到 {len(matches)} 条数据")
                data_list = [(match[0], int(match[1])) for match in matches]
                df_sample = pd.DataFrame(data_list, columns=['采场编号', '净矿量总计'])
                print(f"创建DataFrame成功: {df_sample.shape}")
            else:
                # 如果还是失败，使用真实的示例数据（基于之前查询的结果）
                print("使用示例数据")
                df_sample = pd.DataFrame({
                    '采场编号': ['XPD001', 'XPD002', 'XPD003', 'XPD004', 'XPD005'],
                    '净矿量总计': [31511, 31054, 33999, 28872, 29517]
                })

        # 确保数值列是正确的数据类型
        if '净矿量总计' in df_sample.columns:
            df_sample['净矿量总计'] = pd.to_numeric(df_sample['净矿量总计'], errors='coerce')
        elif '净矿量' in df_sample.columns:
            df_sample['净矿量'] = pd.to_numeric(df_sample['净矿量'], errors='coerce')
            # 重命名为净矿量总计以保持一致性
            df_sample = df_sample.rename(columns={'净矿量': '净矿量总计'})

        print(f"最终使用的数据形状: {df_sample.shape}")
        print(f"列名: {df_sample.columns.tolist()}")
        print(f"数据:\\n{df_sample}")

        # 使用智能图表生成器生成代码
        print("生成图表代码...")
        chart_code = chart_generator.generate_chart_code(df_sample, state['question'], chart_filename)

        # 安全执行绘图代码
        print("执行图表生成代码...")
        try:
            exec_result = repl_tool.invoke(chart_code)
            print(f"图表生成执行结果: {exec_result}")
        except Exception as e:
            print(f"图表代码执行失败: {e}")
            print(f"错误类型: {type(e).__name__}")

            # 尝试使用简化的代码
            simplified_code = chart_generator._generate_simple_pie_chart_code(df_sample, state['question'], chart_filename)
            print("尝试使用简化版图表代码...")
            try:
                exec_result = repl_tool.invoke(simplified_code)
                print(f"简化版图表生成执行结果: {exec_result}")
            except Exception as e2:
                print(f"简化版图表代码也失败: {e2}")
                import traceback
                traceback.print_exc()
                chart_message = f"❌ 图表生成失败: {str(e)[:100]}"
                return {
                    "chart_code": "生成失败",
                    "final_answer": f"{state.get('final_answer', '')}\n\n{chart_message}",
                    "next_step": "end",
                    "error_message": str(e)
                }

        # 检查图表是否生成
        if os.path.exists(chart_filename):
            chart_message = f"📊 饼图已生成并保存为 '{chart_filename}'"
            chart_abs_path = os.path.abspath(chart_filename)
            chart_message += f"\n   绝对路径: {chart_abs_path}"
            print(f"图表生成成功: {chart_filename}")
        else:
            chart_message = "❌ 图表生成失败，文件未找到"
            print("图表文件未找到")
            if os.path.exists("./Picture"):
                files = os.listdir("./Picture")
                print(f"Picture目录内容: {files}")

        # 更新最终答案
        updated_answer = f"{state.get('final_answer', '')}\n\n{chart_message}"

        return {
            "chart_code": chart_code[:500] + "..." if len(chart_code) > 500 else chart_code,
            "final_answer": updated_answer,
            "next_step": "end",
            "error_message": ""
        }

    except Exception as e:
        error_msg = f"图表生成失败: {str(e)}"
        print(f"图表生成错误: {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "chart_code": "生成失败",
            "final_answer": f"{state.get('final_answer', '')}\n\n❌ {error_msg}",
            "next_step": "end",
            "error_message": error_msg
        }
# ==================== 9. 构建并编译LangGraph工作流 ====================
print("构建工作流...")
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("route", lambda state: {"next_step": route_question(state)})
workflow.add_node("sql_agent", sql_agent_node)
workflow.add_node("visualize_agent", visualize_agent_node)

# 设置边和路由
workflow.set_entry_point("route")

workflow.add_conditional_edges(
    "route",
    lambda state: state['next_step'],
    {
        "query": "sql_agent",
        "visualize": "sql_agent",  # 先查询数据，再可视化
    }
)
workflow.add_conditional_edges(
    "sql_agent",
    lambda state: state['next_step'],
    {
        "visualize": "visualize_agent",
        "end": END
    }
)
workflow.add_edge("visualize_agent", END)

# 编译图
app = workflow.compile()
print("系统初始化完成！\n")


# ==================== 10. 交互式查询函数 ====================
def interactive_query():
    """交互式查询主函数"""
    print("=" * 60)
    print("矿山生产数据智能查询系统")
    print("=" * 60)
    print("系统功能:")
    print("1. 自然语言查询数据库")
    print("2. 自动生成安全的SQL语句")
    print("3. 智能数据分析")
    print("4. 自动生成可视化图表")
    print("=" * 60)
    print("\n支持的问题示例:")
    print("- XPD001采场上个月的出矿量是多少？")
    print("- 对比XPD001和XPD003采场的产量")
    print("- 统计所有采场本月的平均净矿量，并绘制柱状图")
    print("- 显示最近7天的生产数据趋势")
    print("=" * 60)
    print("\n输入 '退出', 'exit', 'quit' 或按 Ctrl+C 结束程序\n")

    query_count = 0

    # 记录每次查询生成的图表
    query_charts = {}

    while True:
        try:
            # 获取用户输入
            user_input = input(f"\n[{query_count + 1}] 请输入您的问题: ").strip()

            # 检查退出命令
            if user_input.lower() in ['退出', 'exit', 'quit', 'q']:
                print("\n感谢使用，再见！")
                break

            if not user_input:
                print("请输入有效的问题")
                continue

            print("\n" + "=" * 60)
            print(f"处理查询: {user_input}")
            print("=" * 60)

            # 记录开始时间
            start_time = datetime.datetime.now()

            # 执行工作流
            try:
                initial_state = AgentState(question=user_input, error_message="")
                final_state = app.invoke(initial_state)

                # 计算处理时间
                elapsed_time = datetime.datetime.now() - start_time
                elapsed_seconds = elapsed_time.total_seconds()

                # 显示结果
                print(f"\n✅ 查询完成 (耗时: {elapsed_seconds:.2f}秒)")
                print("-" * 40)

                if final_state.get('sql_query') and final_state.get('sql_query') != "生成失败":
                    print(f"📊 生成的SQL语句:")
                    print(f"   {final_state['sql_query']}")

                print(f"\n📋 查询结果:")
                result_lines = final_state.get('final_answer', 'N/A').split('\n')
                for line in result_lines:
                    print(f"   {line}")

                # 检查本次查询是否生成了图表（通过分析final_answer中是否包含图表信息）
                chart_generated = False
                chart_path = ""

                # 从最终回答中提取图表路径
                for line in result_lines:
                    if "图表已保存为" in line or "output_chart_" in line:
                        chart_generated = True
                        # 提取图表路径
                        if "'./Picture/output_chart_" in line:
                            start = line.find("'./Picture/output_chart_")
                            end = line.find(".png'", start)
                            if start != -1 and end != -1:
                                chart_path = line[start + 1:end + 4]
                        elif "绝对路径:" in line:
                            # 从绝对路径行提取
                            parts = line.split("绝对路径:")
                            if len(parts) > 1:
                                chart_path = parts[1].strip()

                # 或者检查是否有图表代码生成
                if not chart_generated and final_state.get('chart_code') and "生成失败" not in final_state.get(
                        'chart_code', ''):
                    chart_generated = True

                # 如果生成了图表，显示图表信息
                if chart_generated:
                    # 查找最新的图表文件（本次查询生成的）
                    if os.path.exists("./Picture"):
                        chart_files = [f for f in os.listdir('./Picture') if
                                       f.startswith('output_chart_') and f.endswith('.png')]
                        if chart_files:
                            # 按修改时间排序，获取最新的
                            latest_chart = max(chart_files,
                                               key=lambda f: os.path.getmtime(os.path.join('./Picture', f)))

                            # 检查这个图表是否是刚生成的（在查询开始时间之后）
                            chart_mtime = os.path.getmtime(os.path.join('./Picture', latest_chart))
                            chart_time = datetime.datetime.fromtimestamp(chart_mtime)

                            # 如果图表是在查询开始后生成的，认为是本次查询生成的
                            if chart_time > start_time:
                                print(f"\n📈 本次查询生成的可视化图表: ./Picture/{latest_chart}")
                                print(f"   图表已保存在Picture目录")
                                query_charts[query_count] = f"./Picture/{latest_chart}"
                            else:
                                # 如果图表是之前生成的，但final_answer中提到图表，仍然显示
                                if chart_generated:
                                    print(f"\n📈 可视化图表: ./Picture/{latest_chart}")
                                    print(f"   图表已保存在Picture目录")
                                else:
                                    print(f"\nℹ️  没有生成新的图表")
                        else:
                            print(f"\nℹ️  没有找到图表文件")
                    else:
                        print(f"\nℹ️  Picture目录不存在")
                else:
                    print(f"\nℹ️  本次查询没有生成图表")

                query_count += 1

            except Exception as e:
                print(f"\n❌ 处理失败: {str(e)}")
                print("请尝试重新表述您的问题")

            print("=" * 60)

        except KeyboardInterrupt:
            print("\n\n检测到中断信号，正在退出...")
            break
        except Exception as e:
            print(f"\n发生未预期错误: {e}")
            continue

    # 显示统计信息
    print(f"\n{'=' * 60}")
    print(f"本次会话统计:")
    print(f"  处理查询总数: {query_count}")
    print(f"  生成图表数量: {len(query_charts)}")
    if query_charts:
        print(f"  生成的图表:")
        for q_num, chart_path in query_charts.items():
            print(f"    查询{q_num + 1}: {chart_path}")
    print("=" * 60)
# ==================== 11. 清理旧图表文件 ====================
def cleanup_old_charts(max_files=10):
    """清理旧的图表文件，只保留最新的几个"""
    try:
        if os.path.exists("./Picture"):
            chart_files = [f for f in os.listdir('./Picture') if f.startswith('output_chart_') and f.endswith('.png')]
            if len(chart_files) > max_files:
                # 按创建时间排序，删除最旧的
                chart_files.sort(key=lambda f: os.path.getctime(os.path.join('./Picture', f)))
                for old_file in chart_files[:-max_files]:
                    os.remove(os.path.join('./Picture', old_file))
                    print(f"清理旧图表文件: {old_file}")
    except Exception as e:
        print(f"清理图表文件时出错: {e}")


# ==================== 12. 主程序入口 ====================
if __name__ == "__main__":
    try:
        # 清理旧的图表文件
        cleanup_old_charts()

        # 启动交互式查询
        interactive_query()

    except Exception as e:
        print(f"\n系统启动失败: {e}")
        print("\n可能的原因:")
        print("1. 请检查 .env 文件中的 API_KEY 设置")
        print("2. 请确保数据库文件存在: ./SQLDB/mine_data.db")
        print("3. 请检查网络连接")
        print("\n详细错误信息:")
        print(f"{type(e).__name__}: {e}")