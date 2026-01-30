import datetime
import os
from datetime import time

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
# ==================== 1. 初始化模型与数据库 ====================
# 初始化DeepSeek模型[citation:2][citation:10]
print("正在初始化系统...")

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    API_KEY = os.getenv("DEEPSEEK_API_KEY")  # 也尝试使用 DEEPSEEK_API_KEY

if not API_KEY:
    print("错误: 未找到API密钥！请在.env文件中设置 API_KEY 或 DEEPSEEK_API_KEY")
    print("格式: API_KEY=your_deepseek_api_key_here")
    exit(1)

llm = ChatOpenAI(
    model="deepseek-chat",  # 注意: 新版使用 model 而不是 model_name
    base_url="https://api.deepseek.com/v1",  # 使用 base_url 而不是 openai_api_base
    api_key=API_KEY,  # 使用 api_key 而不是 openai_api_key
    temperature=0.1,
    max_tokens=2048
)

# 测试数据库连接
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


# ==================== 2. 定义智能体状态 ====================
class AgentState(TypedDict):
    question: str
    sql_query: str
    query_result: str
    chart_code: str
    final_answer: str
    next_step: str


# ==================== 3. 构建智能体工具 ====================
print("正在初始化工具...")
try:
    write_query_tool = create_sql_query_chain(llm, db)
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    repl_tool = PythonAstREPLTool(locals={"pd": pd, "plt": plt, "sns": sns})
    print("工具初始化完成！")
except Exception as e:
    print(f"工具初始化失败: {e}")
    exit(1)


# ==================== 4. 定义路由与节点函数 ====================
def route_question(state: AgentState):
    """核心路由逻辑：判断用户意图是查询数据还是分析绘图"""
    question = state['question'].lower()

    # 增强的关键词识别
    visualize_keywords = ['统计', '趋势', '图表', '画图', '可视化', '分析', '柱状图',
                          '折线图', '饼图', '分布', '对比', '比较', '展示']
    query_keywords = ['查询', '查看', '显示', '获取', '多少', '何时', '哪里', '谁']

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

    try:
        # 1. 生成SQL
        sql_query = write_query_tool.invoke({"question": state['question']})
        print(f"生成的SQL: {sql_query}")

        # 2. 执行查询
        query_result = execute_query_tool.invoke(sql_query)

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

        # 4. 判断是否需要可视化
        result_str = str(query_result)
        needs_visualization = (
                "失败" not in result_str and
                len(result_str.split('\n')) > 3 and
                any(field in state['question'] for field in ['趋势', '对比', '统计', '图表', '画图'])
        )

        return {
            "sql_query": sql_query,
            "query_result": str(query_result)[:1000],
            "final_answer": final_answer,
            "next_step": "visualize" if needs_visualization else "end"
        }

    except Exception as e:
        error_msg = f"SQL处理失败: {str(e)}"
        return {
            "sql_query": "生成失败",
            "query_result": error_msg,
            "final_answer": f"抱歉，处理查询时出错: {str(e)[:100]}",
            "next_step": "end"
        }


def visualize_agent_node(state: AgentState):
    """可视化智能体节点：生成分析图表"""
    print("生成可视化图表...")

    try:
        # 生成带时间戳的唯一文件名
        timestamp = int(datetime.datetime.now().timestamp())
        chart_filename = f"./Picture/output_chart_{timestamp}.png"

        # 构建绘图提示词
        visualize_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个Python数据分析专家。根据用户问题和数据摘要，生成合适的Matplotlib或Seaborn绘图代码。

要求：
1. 代码必须将图形保存为 '{chart_filename}'
2. 添加清晰的中文标题、坐标轴标签
3. 根据数据特点选择最合适的图表类型
4. 确保图表美观、可读性好
5. 只输出Python代码，不要额外解释

数据摘要：
{data_sample}

用户问题：{question}

生成的Python代码：""")
        ])

        # 准备数据
        data_sample = state.get('query_result', '暂无数据')

        # 生成绘图代码
        code_chain = visualize_prompt | llm | StrOutputParser()
        chart_code = code_chain.invoke({
            "data_sample": str(data_sample)[:800],
            "question": state['question'],
            "chart_filename": chart_filename
        })

        # 安全执行绘图代码
        print("执行图表生成代码...")
        exec_result = repl_tool.invoke(chart_code)

        # 检查图表是否生成
        if os.path.exists(chart_filename):
            chart_message = f"图表已生成并保存为 '{chart_filename}'"
        else:
            chart_message = "图表生成失败，但代码执行未报错"

        # 更新最终答案
        updated_answer = f"{state.get('final_answer', '')}\n\n📊 {chart_message}"

        return {
            "chart_code": chart_code,
            "final_answer": updated_answer,
            "next_step": "end"
        }

    except Exception as e:
        error_msg = f"图表生成失败: {str(e)}"
        return {
            "chart_code": "生成失败",
            "final_answer": f"{state.get('final_answer', '')}\n\n❌ {error_msg}",
            "next_step": "end"
        }


# ==================== 5. 构建并编译LangGraph工作流 ====================
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


# ==================== 6. 交互式查询函数 ====================
def interactive_query():
    """交互式查询主函数"""
    print("=" * 60)
    print("矿山生产数据智能查询系统")
    print("=" * 60)
    print("系统功能:")
    print("1. 自然语言查询数据库")
    print("2. 自动生成SQL语句")
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
            start_time = time.time()

            # 执行工作流
            try:
                initial_state = AgentState(question=user_input)
                final_state = app.invoke(initial_state)

                # 计算处理时间
                elapsed_time = time.time() - start_time

                # 显示结果
                print(f"\n✅ 查询完成 (耗时: {elapsed_time:.2f}秒)")
                print("-" * 40)

                if final_state.get('sql_query') and final_state.get('sql_query') != "生成失败":
                    print(f"📊 生成的SQL语句:")
                    print(f"   {final_state['sql_query']}")

                print(f"\n📋 查询结果:")
                print(f"   {final_state.get('final_answer', 'N/A')}")

                # 检查是否有图表生成
                chart_files = [f for f in os.listdir('.') if f.startswith('output_chart_') and f.endswith('.png')]
                if chart_files:
                    latest_chart = max(chart_files, key=os.path.getctime)
                    print(f"\n📈 可视化图表: {latest_chart}")
                    print("   图表已保存在当前目录")

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
    print(f"  生成图表数量: {len([f for f in os.listdir('.') if f.startswith('output_chart_')])}")
    print("=" * 60)


# ==================== 7. 清理旧图表文件 ====================
def cleanup_old_charts(max_files=10):
    """清理旧的图表文件，只保留最新的几个"""
    try:
        chart_files = [f for f in os.listdir('.') if f.startswith('output_chart_') and f.endswith('.png')]
        if len(chart_files) > max_files:
            # 按创建时间排序，删除最旧的
            chart_files.sort(key=os.path.getctime)
            for old_file in chart_files[:-max_files]:
                os.remove(old_file)
                print(f"清理旧图表文件: {old_file}")
    except:
        pass  # 忽略清理错误


# ==================== 8. 主程序入口 ====================
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