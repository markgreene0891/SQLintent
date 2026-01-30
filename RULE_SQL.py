import json
import re
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class SQLIntentRecognizer:
    def __init__(self, config_path: str = None):
        # 查询关键词
        self.query_keywords = [
            '查询', '查找', '搜索', '显示', '查看', '获取', '查', '搜',
            'select', 'query', 'find', 'show', 'get',
            '对比', '比较', '分析', '导出', '列出'
        ]

        # 禁止的操作关键词
        self.forbidden_keywords = [
            'insert', 'update', 'delete', 'drop', 'alter',
            'create', 'truncate', 'grant', 'revoke'
        ]

        # 加载配置文件
        self.config = self._load_config(config_path) if config_path else self._create_default_config()

        # 构建字段别名反向映射
        self._build_field_mappings()

        # 统计函数关键词
        self.stat_keywords = ['统计', '计算', '汇总', '总数', '总和', '平均', '最大值', '最小值', '总量', '合计']

        # 时间相关关键词
        self.time_keywords = ['今天', '昨天', '本周', '本月', '今年', '最近', '上周', '上月', '去年', '上个月',
                              '前个月']

        # 对比关键词
        self.compare_keywords = ['对比', '比较', '对比分析', '对比查询', '比较分析']

        # 数据库类型，默认为SQLite
        self.db_type = 'sqlite'  # 可以设置为 'mysql' 或 'postgresql'

        # 数值字段列表（用于统计查询）
        self.numeric_fields = ['出矿量', '废石量', '净矿量']

        # 数据库连接
        self.conn = None

    def _load_config(self, config_path: str) -> Dict:
        """加载JSON配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _create_default_config(self) -> Dict:
        """创建默认配置"""
        return {
            "tables": {
                "mine_production": {
                    "name_cn": "采场生产数据",
                    "fields": {
                        "采场编号": ["采场编号", "采场", "矿场", "场区"],
                        "日期": ["日期", "时间", "生产日期"],
                        "出矿量": ["出矿量", "产量", "矿石产量", "矿量", "粗矿量", "出矿", "出矿总量", "总出矿量"],
                        "废石量": ["废石量", "废料", "废石", "废石产出"],
                        "净矿量": ["净矿量", "净产量", "净出矿量"]
                    },
                    "common_queries": [
                        {
                            "pattern": "(XPD\\d{3})",
                            "template": "SELECT * FROM mine_production WHERE 采场编号 = '{0}'"
                        },
                        {
                            "pattern": "最近(\\d+)天",
                            "template": "SELECT * FROM mine_production WHERE 日期 >= date('now', '-{0} days')"
                        }
                    ]
                }
            },
            "patterns": {
                "simple_query": "(查询|查看|显示|获取|分析|导出|列出)\\s*(.*)",
                "stat_query": "(统计|计算|汇总)\\s*(.*?)\\s*(数量|平均|总和|最大值|最小值)",
                "compare_query": "(对比|比较|对比分析)\\s*(.*?)\\s*(和|与|以及|及)\\s*(.*)"
            }
        }

    def _build_field_mappings(self):
        """构建字段别名到实际字段名的映射"""
        self.field_mapping = {}
        self.cn_to_field = {}

        for table_name, table_info in self.config['tables'].items():
            for field_name, aliases in table_info['fields'].items():
                # 字段名到中文别名的映射
                self.field_mapping[field_name] = aliases

                # 中文别名到字段名的反向映射
                for alias in aliases:
                    self.cn_to_field[alias] = (table_name, field_name)

    def set_db_type(self, db_type: str):
        """设置数据库类型：sqlite, mysql, postgresql"""
        self.db_type = db_type

    def connect_db(self, db_path: str = "./SQLDB/mine_data.db"):
        """连接SQLite数据库"""
        try:
            self.conn = sqlite3.connect(db_path)
            return True
        except Exception as e:
            print(f"数据库连接失败: {e}")
            return False

    def disconnect_db(self):
        """断开数据库连接"""
        if self.conn:
            self.conn.close()

    def execute_query(self, sql: str) -> Tuple[bool, Any, List[str]]:
        """
        执行SQL查询
        返回: (是否成功, 查询结果, 字段名列表)
        """
        if not self.conn:
            return False, None, []

        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)

            # 获取字段名
            field_names = [description[0] for description in cursor.description]

            # 获取所有结果
            result = cursor.fetchall()

            return True, result, field_names
        except Exception as e:
            return False, f"查询执行失败: {e}", []

    def execute_query_to_dataframe(self, sql: str) -> Tuple[bool, Any]:
        """
        执行SQL查询并返回DataFrame
        返回: (是否成功, DataFrame或错误信息)
        """
        if not self.conn:
            return False, "数据库未连接"

        try:
            df = pd.read_sql_query(sql, self.conn)
            return True, df
        except Exception as e:
            return False, f"查询执行失败: {e}"

    def _extract_time_condition(self, user_input: str) -> Optional[str]:
        """提取时间条件（根据数据库类型生成不同的SQL）"""
        # SQLite版本
        sqlite_conditions = {
            '今天': "日期 = date('now')",
            '昨天': "日期 = date('now', '-1 day')",
            '本周': "strftime('%W', 日期) = strftime('%W', 'now') AND strftime('%Y', 日期) = strftime('%Y', 'now')",
            '本月': "strftime('%m', 日期) = strftime('%m', 'now') AND strftime('%Y', 日期) = strftime('%Y', 'now')",
            '今年': "strftime('%Y', 日期) = strftime('%Y', 'now')",
            '最近7天': "日期 >= date('now', '-7 days')",
            '最近30天': "日期 >= date('now', '-30 days')",
            '最近90天': "日期 >= date('now', '-90 days')",
            '上个月': "strftime('%m', 日期) = strftime('%m', date('now', '-1 month')) AND strftime('%Y', 日期) = strftime('%Y', date('now', '-1 month'))",
        }

        # 检查固定时间条件
        for chinese in sqlite_conditions.keys():
            if chinese in user_input:
                if self.db_type == 'sqlite':
                    return sqlite_conditions[chinese]

        # 匹配"最近X天"
        recent_days_match = re.search(r'最近(\d+)天', user_input)
        if recent_days_match:
            days = recent_days_match.group(1)
            if self.db_type == 'sqlite':
                return f"日期 >= date('now', '-{days} days')"

        # 匹配具体年份和月份
        year_month_match = re.search(r'(\d{4})年(\d{1,2})月', user_input)
        if year_month_match:
            year = year_month_match.group(1)
            month = year_month_match.group(2).zfill(2)
            next_month = str(int(month) + 1).zfill(2) if int(month) < 12 else '01'
            next_year = year if int(month) < 12 else str(int(year) + 1)

            return f"日期 >= '{year}-{month}-01' AND 日期 < '{next_year}-{next_month}-01'"

        return None

    def is_direct_sql(self, user_input: str) -> bool:
        """检测是否直接输入SQL语句"""
        sql_patterns = [
            r'^\s*SELECT\s', r'^\s*FROM\s', r'\sWHERE\s',
            r'\sJOIN\s', r'\sGROUP BY\s', r'\sORDER BY\s',
            r'^\s*INSERT\s', r'^\s*UPDATE\s', r'^\s*DELETE\s',
            r'\sSET\s', r'\sVALUES\s'
        ]

        input_upper = user_input.upper()
        count = sum(1 for pattern in sql_patterns if re.search(pattern, input_upper))
        return count >= 2  # 包含两个以上SQL关键字

    def contains_forbidden_keywords(self, text: str) -> bool:
        """检查是否包含禁止的操作"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.forbidden_keywords)

    def is_query_intent(self, user_input: str) -> bool:
        """判断是否是查询意图（更宽松的判断）"""
        input_lower = user_input.lower()

        # 检查是否包含任何查询关键词
        if any(keyword in input_lower for keyword in self.query_keywords):
            return True

        # 检查是否包含统计关键词
        if any(keyword in user_input for keyword in self.stat_keywords):
            return True

        # 检查是否包含对比关键词
        if any(keyword in user_input for keyword in self.compare_keywords):
            return True

        # 检查是否包含时间关键词（可能是查询意图）
        if any(keyword in user_input for keyword in self.time_keywords):
            return True

        # 检查是否包含采场编号（可能是查询意图）
        if re.search(r'(XPD\d{3})', user_input):
            return True

        return False

    def classify_intent(self, user_input: str) -> Dict[str, Any]:
        """分类用户意图"""
        intent = {
            "type": "unknown",
            "table": None,
            "fields": [],
            "conditions": [],
            "group_by": None,
            "order_by": None,
            "limit": None,
            "compare_items": []  # 新增：对比项
        }

        # 1. 检查是否是对比查询
        if any(keyword in user_input for keyword in self.compare_keywords):
            intent["type"] = "comparison"
            return self._parse_compare_query(user_input, intent)

        # 2. 检查是否是统计查询
        if any(keyword in user_input for keyword in self.stat_keywords):
            intent["type"] = "statistical"
            return self._parse_statistical_query(user_input, intent)

        # 3. 检查是否匹配预定义模式
        for pattern_name, pattern in self.config['patterns'].items():
            match = re.search(pattern, user_input)
            if match:
                if pattern_name == "simple_query":
                    intent["type"] = "simple"
                    return self._parse_simple_query(user_input, intent, match)
                elif pattern_name == "stat_query":
                    intent["type"] = "statistical"
                    return self._parse_stat_query(user_input, intent, match)
                elif pattern_name == "compare_query":
                    intent["type"] = "comparison"
                    return self._parse_compare_query(user_input, intent, match)

        # 4. 默认作为简单查询处理
        intent["type"] = "simple"
        return self._parse_simple_query(user_input, intent, None)

    def _parse_compare_query(self, user_input: str, intent: Dict, match: Optional[re.Match] = None) -> Dict:
        """解析对比查询"""
        intent["table"] = self._detect_table(user_input)

        # 提取对比的采场编号
        pit_ids = re.findall(r'(XPD\d{3})', user_input)
        if pit_ids:
            intent["compare_items"] = pit_ids

            # 如果只有两个采场，创建对比条件
            if len(pit_ids) == 2:
                conditions = [
                    f"采场编号 = '{pit_ids[0]}' OR 采场编号 = '{pit_ids[1]}'"
                ]
                intent["conditions"] = conditions
                intent["order_by"] = "采场编号, 日期"  # 按采场和日期排序，便于对比
            else:
                # 多个采场，使用IN条件
                pit_list = ", ".join([f"'{pid}'" for pid in pit_ids])
                conditions = [f"采场编号 IN ({pit_list})"]
                intent["conditions"] = conditions
                intent["order_by"] = "采场编号, 日期"

        # 提取查询字段
        if '产量' in user_input:
            # 如果是产量对比，需要出矿量、净矿量等
            intent["fields"] = ['采场编号', '日期', '出矿量', '净矿量']
        else:
            intent["fields"] = self._extract_fields(user_input, intent["table"])

        # 提取时间范围
        time_condition = self._extract_time_condition(user_input)
        if time_condition:
            if intent["conditions"]:
                intent["conditions"].append(time_condition)
            else:
                intent["conditions"] = [time_condition]

        return intent

    def _parse_simple_query(self, user_input: str, intent: Dict, match: Optional[re.Match]) -> Dict:
        """解析简单查询"""
        # 确定查询的表
        intent["table"] = self._detect_table(user_input)

        # 提取查询字段
        intent["fields"] = self._extract_fields(user_input, intent["table"])

        # 提取查询条件
        intent["conditions"] = self._extract_conditions(user_input, intent["table"])

        # 提取排序
        intent["order_by"] = self._extract_order_by(user_input)

        # 提取限制数量
        intent["limit"] = self._extract_limit(user_input)

        return intent

    def _parse_statistical_query(self, user_input: str, intent: Dict) -> Dict:
        """解析统计查询"""
        intent["table"] = self._detect_table(user_input)

        # 统计类型
        if '平均' in user_input:
            intent["stat_type"] = "avg"
        elif '总和' in user_input or '总数' in user_input or '总量' in user_input or '合计' in user_input:
            intent["stat_type"] = "sum"
        elif '最大' in user_input:
            intent["stat_type"] = "max"
        elif '最小' in user_input:
            intent["stat_type"] = "min"
        else:
            intent["stat_type"] = "count"

        # 统计字段 - 关键修复：确保统计查询有具体的字段
        fields = self._extract_fields(user_input, intent["table"])
        if not fields:
            # 如果没指定字段，根据上下文推断
            if '产量' in user_input or '出矿' in user_input:
                fields = ['出矿量']
            elif '废石' in user_input:
                fields = ['废石量']
            elif '净矿' in user_input:
                fields = ['净矿量']
            else:
                # 默认使用出矿量
                fields = ['出矿量']

        # 确保字段不是通配符
        if fields == ["*"]:
            fields = ['出矿量']  # 默认使用出矿量

        intent["fields"] = fields

        # 分组字段
        if '按' in user_input:
            group_field_match = re.search(r'按(.+?)(统计|计算|汇总|分组)', user_input)
            if group_field_match:
                group_field = group_field_match.group(1).strip()
                field_name = self._map_cn_to_field(group_field, intent["table"])
                if field_name:
                    intent["group_by"] = field_name

        intent["conditions"] = self._extract_conditions(user_input, intent["table"])

        return intent

    def _parse_stat_query(self, user_input: str, intent: Dict, match: re.Match) -> Dict:
        """解析统计查询（模式匹配版）"""
        return self._parse_statistical_query(user_input, intent)

    def _detect_table(self, user_input: str) -> str:
        """检测查询的表"""
        # 根据关键词匹配表
        for table_name, table_info in self.config['tables'].items():
            name_cn = table_info.get('name_cn', '')
            if name_cn and name_cn in user_input:
                return table_name

            # 检查字段是否出现
            for field_aliases in table_info['fields'].values():
                for alias in field_aliases:
                    if alias in user_input:
                        return table_name

        # 默认返回第一个表
        return list(self.config['tables'].keys())[0]

    def _extract_fields(self, user_input: str, table_name: str) -> List[str]:
        """提取查询字段"""
        fields = []

        # 检查用户是否指定了特定字段
        for cn_alias, (tbl_name, field_name) in self.cn_to_field.items():
            if tbl_name == table_name and cn_alias in user_input:
                fields.append(field_name)

        # 如果用户说"所有"或"全部"或未指定字段，返回空列表（表示SELECT *）
        if '所有' in user_input or '全部' in user_input or '所有数据' in user_input:
            return []

        # 如果用户说"详细信息"，返回所有字段
        if '详细' in user_input:
            table_info = self.config['tables'][table_name]
            return list(table_info['fields'].keys())

        return list(set(fields)) if fields else []

    def _extract_conditions(self, user_input: str, table_name: str) -> List[str]:
        """提取查询条件"""
        conditions = []

        # 1. 先检查常见查询模式
        table_info = self.config['tables'][table_name]
        if 'common_queries' in table_info:
            for query in table_info['common_queries']:
                pattern = query['pattern']
                match = re.search(pattern, user_input)
                if match:
                    # 这里简化处理，实际可能需要根据模板生成条件
                    template = query['template']
                    # 提取匹配组并应用到模板
                    groups = match.groups()
                    if groups:
                        condition = template.format(*groups)
                        # 从模板中提取WHERE条件部分
                        if 'WHERE' in condition:
                            where_part = condition.split('WHERE')[1].strip()
                            conditions.append(where_part)

        # 2. 时间条件
        time_condition = self._extract_time_condition(user_input)
        if time_condition:
            conditions.append(time_condition)

        # 3. 采场编号条件
        pit_id_match = re.search(r'(XPD\d{3})', user_input)
        if pit_id_match and not any(keyword in user_input for keyword in self.compare_keywords):
            # 如果不是对比查询，才添加单个采场条件
            conditions.append(f"采场编号 = '{pit_id_match.group(1)}'")

        # 4. 数值范围条件
        range_patterns = [
            (r'大于(\d+)', '> {0}'),
            (r'小于(\d+)', '< {0}'),
            (r'超过(\d+)', '> {0}'),
            (r'低于(\d+)', '< {0}'),
            (r'等于(\d+)', '= {0}'),
            (r'不低于(\d+)', '>= {0}'),
            (r'不高于(\d+)', '<= {0}'),
        ]

        for pattern, operator in range_patterns:
            match = re.search(pattern, user_input)
            if match:
                value = match.group(1)
                # 确定是哪个数值字段
                for field in self.numeric_fields:
                    for alias in self.field_mapping.get(field, []):
                        if alias in user_input:
                            conditions.append(f"{field} {operator.format(value)}")
                            break

        return conditions

    def _extract_order_by(self, user_input: str) -> Optional[str]:
        """提取排序条件"""
        if '最新' in user_input or '最近' in user_input:
            return "日期 DESC"
        elif '最早' in user_input:
            return "日期 ASC"
        elif '最多' in user_input and ('出矿' in user_input or '产量' in user_input):
            return "出矿量 DESC"
        elif '最少' in user_input and ('出矿' in user_input or '产量' in user_input):
            return "出矿量 ASC"
        elif '最高' in user_input:
            return "出矿量 DESC"
        elif '最低' in user_input:
            return "出矿量 ASC"
        elif '降序' in user_input:
            # 找到要排序的字段
            for field in self.numeric_fields:
                for alias in self.field_mapping.get(field, []):
                    if alias in user_input:
                        return f"{field} DESC"
        elif '升序' in user_input:
            for field in self.numeric_fields:
                for alias in self.field_mapping.get(field, []):
                    if alias in user_input:
                        return f"{field} ASC"
        return None

    def _extract_limit(self, user_input: str) -> Optional[int]:
        """提取限制数量"""
        # 匹配数字
        numbers = re.findall(r'(\d+)(个|条|项|天)?', user_input)
        if numbers:
            return int(numbers[0][0])

        # 匹配常见数量词
        limit_words = {
            '前10': 10, '前十': 10,
            '前5': 5, '前五': 5,
            '前20': 20, '前二十': 20,
            '前100': 100, '前一百': 100,
            '最近几条': 10, '最新几条': 10,
            '所有': None, '全部': None,
        }

        for word, limit in limit_words.items():
            if word in user_input:
                return limit

        return None

    def _map_cn_to_field(self, cn_text: str, table_name: str) -> Optional[str]:
        """将中文映射到字段名"""
        for cn_alias, (tbl_name, field_name) in self.cn_to_field.items():
            if tbl_name == table_name and cn_alias in cn_text:
                return field_name
        return None

    def build_sql(self, intent: Dict) -> str:
        """根据意图构建SQL"""
        table = intent["table"]
        fields = intent["fields"]
        conditions = intent["conditions"]
        group_by = intent.get("group_by")
        order_by = intent.get("order_by")
        limit = intent.get("limit")

        # 构建SELECT部分
        if intent["type"] == "statistical":
            stat_type = intent.get("stat_type", "count")

            # 关键修复：确保统计函数有具体的字段
            if fields and fields != ["*"]:
                field_list = ', '.join(fields)
                if stat_type == "count":
                    select_clause = f"SELECT COUNT(*) as 总数"
                elif stat_type == "sum":
                    select_clause = f"SELECT SUM({field_list}) as 总和"
                elif stat_type == "avg":
                    select_clause = f"SELECT AVG({field_list}) as 平均值"
                elif stat_type == "max":
                    select_clause = f"SELECT MAX({field_list}) as 最大值"
                elif stat_type == "min":
                    select_clause = f"SELECT MIN({field_list}) as 最小值"
            else:
                # 如果没有指定字段，对于统计查询需要指定默认字段
                if stat_type == "count":
                    select_clause = f"SELECT COUNT(*) as 总数"
                else:
                    # 对于其他统计函数，默认使用出矿量
                    default_field = "出矿量"
                    if stat_type == "sum":
                        select_clause = f"SELECT SUM({default_field}) as 总和"
                    elif stat_type == "avg":
                        select_clause = f"SELECT AVG({default_field}) as 平均值"
                    elif stat_type == "max":
                        select_clause = f"SELECT MAX({default_field}) as 最大值"
                    elif stat_type == "min":
                        select_clause = f"SELECT MIN({default_field}) as 最小值"
        elif intent["type"] == "comparison":
            # 对比查询的特殊处理
            if fields:
                select_clause = f"SELECT {', '.join(fields)}"
            else:
                select_clause = "SELECT *"
        else:
            if fields:
                select_clause = f"SELECT {', '.join(fields)}"
            else:
                select_clause = "SELECT *"

        # 构建FROM部分
        from_clause = f"FROM {table}"

        sql_parts = [select_clause, from_clause]

        # 构建WHERE部分
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
            sql_parts.append(where_clause)

        # 构建GROUP BY部分
        if group_by:
            sql_parts.append(f"GROUP BY {group_by}")

        # 构建ORDER BY部分
        if order_by:
            sql_parts.append(f"ORDER BY {order_by}")

        # 构建LIMIT部分
        if limit is not None:
            sql_parts.append(f"LIMIT {limit}")

        return " ".join(sql_parts)

    def process_input(self, user_input: str) -> Tuple[bool, Optional[str]]:
        """
        处理用户输入
        返回: (是否允许, SQL语句或错误信息)
        """
        # 1. 检查是否直接输入SQL
        if self.is_direct_sql(user_input):
            return False, "请使用自然语言描述您的查询需求"

        # 2. 检查是否包含禁止的操作
        if self.contains_forbidden_keywords(user_input):
            return False, "只支持查询操作"

        # 3. 检查是否是查询意图（使用更宽松的判断）
        if not self.is_query_intent(user_input):
            return False, "请描述您的查询需求"

        try:
            # 4. 分类意图并生成SQL
            intent = self.classify_intent(user_input)
            sql = self.build_sql(intent)
            return True, sql
        except Exception as e:
            return False, f"无法理解您的查询需求: {str(e)}"

    def analyze_and_visualize(self, df: pd.DataFrame, query_text: str, save_path: str = None) -> plt.Figure:
        """
        分析查询结果并生成可视化图表

        参数:
            df: 查询结果的DataFrame
            query_text: 原始查询文本
            save_path: 图表保存路径（可选）

        返回:
            matplotlib Figure对象
        """
        if df.empty:
            print("查询结果为空，无法生成图表")
            return None

        # 根据查询类型和数据结构选择图表类型
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'查询分析: {query_text}', fontsize=16, fontweight='bold')

        # 1. 数据概览（表格）
        ax1 = axes[0, 0]
        ax1.axis('tight')
        ax1.axis('off')

        # 创建表格
        table_data = [df.columns.tolist()] + df.head(10).values.tolist()
        table = ax1.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax1.set_title('数据概览 (前10行)')

        # 2. 根据数据类型生成不同的图表

        # 检查是否有日期字段
        has_date = '日期' in df.columns
        has_pit_id = '采场编号' in df.columns
        has_numeric = any(col in self.numeric_fields for col in df.columns)

        # 如果有日期和数值字段，生成时间序列图
        if has_date and has_numeric:
            ax2 = axes[0, 1]

            # 尝试将日期列转换为datetime
            try:
                df['日期'] = pd.to_datetime(df['日期'])
            except:
                pass

            # 获取数值列
            numeric_cols = [col for col in df.columns if col in self.numeric_fields]

            if has_pit_id and len(df['采场编号'].unique()) <= 5:
                # 按采场分组绘制多条线
                for pit_id in df['采场编号'].unique():
                    pit_data = df[df['采场编号'] == pit_id]
                    for col in numeric_cols:
                        if col in pit_data.columns:
                            ax2.plot(pit_data['日期'], pit_data[col], marker='o', label=f'{pit_id}-{col}')
                ax2.set_xlabel('日期')
                ax2.set_ylabel('数值')
                ax2.set_title('按采场时间序列图')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                # 绘制所有数据的趋势
                for col in numeric_cols:
                    if col in df.columns:
                        ax2.plot(df['日期'], df[col], marker='o', label=col)
                ax2.set_xlabel('日期')
                ax2.set_ylabel('数值')
                ax2.set_title('时间序列图')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

        # 如果有采场编号和数值字段，生成柱状图
        elif has_pit_id and has_numeric:
            ax2 = axes[0, 1]

            # 获取数值列
            numeric_cols = [col for col in df.columns if col in self.numeric_fields]

            if len(numeric_cols) == 1:
                # 单数值字段，简单柱状图
                col = numeric_cols[0]
                pit_stats = df.groupby('采场编号')[col].sum()
                ax2.bar(pit_stats.index, pit_stats.values)
                ax2.set_xlabel('采场编号')
                ax2.set_ylabel(col)
                ax2.set_title(f'各采场{col}统计')
                ax2.grid(True, alpha=0.3, axis='y')
            else:
                # 多数值字段，分组柱状图
                x = np.arange(len(df['采场编号'].unique()))
                width = 0.8 / len(numeric_cols)

                for i, col in enumerate(numeric_cols):
                    offset = (i - len(numeric_cols) / 2 + 0.5) * width
                    ax2.bar(x + offset, df[col], width=width, label=col)

                ax2.set_xlabel('采场编号')
                ax2.set_ylabel('数值')
                ax2.set_title('多指标对比图')
                ax2.set_xticks(x)
                ax2.set_xticklabels(df['采场编号'])
                ax2.legend()
                ax2.grid(True, alpha=0.3, axis='y')

        # 3. 数值字段的分布图
        ax3 = axes[1, 0]
        numeric_cols = [col for col in df.columns if col in self.numeric_fields]

        if numeric_cols:
            # 选择第一个数值字段进行分布分析
            col = numeric_cols[0]
            ax3.hist(df[col].dropna(), bins=20, alpha=0.7, edgecolor='black')
            ax3.set_xlabel(col)
            ax3.set_ylabel('频次')
            ax3.set_title(f'{col}分布直方图')
            ax3.grid(True, alpha=0.3)

        # 4. 相关性或汇总统计
        ax4 = axes[1, 1]

        if len(numeric_cols) >= 2:
            # 散点图展示两个数值字段的关系
            x_col = numeric_cols[0]
            y_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
            ax4.scatter(df[x_col], df[y_col], alpha=0.6)
            ax4.set_xlabel(x_col)
            ax4.set_ylabel(y_col)
            ax4.set_title(f'{x_col} vs {y_col} 散点图')
            ax4.grid(True, alpha=0.3)
        elif numeric_cols:
            # 箱线图展示数值字段的统计信息
            col = numeric_cols[0]
            ax4.boxplot(df[col].dropna())
            ax4.set_ylabel(col)
            ax4.set_title(f'{col}箱线图')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")

        return fig

    def process_query_with_visualization(self, user_input: str, save_chart: bool = False) -> Dict[str, Any]:
        """
        完整处理流程：意图识别 -> SQL生成 -> 执行查询 -> 可视化

        参数:
            user_input: 用户自然语言查询
            save_chart: 是否保存图表

        返回:
            包含所有结果的字典
        """
        result = {
            "user_input": user_input,
            "sql_generated": None,
            "query_success": False,
            "dataframe": None,
            "row_count": 0,
            "figure": None
        }

        # 1. 意图识别和SQL生成
        allowed, sql = self.process_input(user_input)
        if not allowed:
            result["error"] = sql
            return result

        result["sql_generated"] = sql

        # 2. 执行查询
        if not self.conn:
            # 如果没有连接，尝试连接默认数据库
            self.connect_db()

        if self.conn:
            success, df = self.execute_query_to_dataframe(sql)
            if success:
                result["query_success"] = True
                result["dataframe"] = df
                result["row_count"] = len(df)

                # 3. 显示查询结果
                print(f"\n{'=' * 60}")
                print(f"用户查询: {user_input}")
                print(f"生成的SQL: {sql}")
                print(f"查询结果: {len(df)} 条记录")
                print(f"{'=' * 60}\n")

                # 显示前几行数据
                if not df.empty:
                    print("查询结果预览:")
                    print(df.head())

                    # 4. 生成可视化图表
                    if len(df) > 0:
                        chart_path = f"./Picture/query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png" if save_chart else None
                        fig = self.analyze_and_visualize(df, user_input, chart_path)
                        if fig:
                            result["figure"] = fig
                            # 显示图表
                            plt.show()
            else:
                result["error"] = df
        else:
            result["error"] = "数据库连接失败"

        return result


# 使用示例
def main():
    # 创建识别器
    recognizer = SQLIntentRecognizer()
    recognizer.set_db_type('sqlite')

    # 连接数据库
    if recognizer.connect_db("./SQLDB/mine_data.db"):
        print("数据库连接成功！")

        # 测试查询
        test_queries = [
            "查询最近7天的出矿量",
            "显示XPD001采场的数据",
            "统计所有采场的出矿总量",
            "对比XPD001和XPD003采场的产量",
            "查看2025年12月的生产数据",
            "显示出矿量大于1000吨的记录",
        ]

        for query in test_queries:
            print(f"\n{'=' * 60}")
            print(f"处理查询: {query}")
            print('=' * 60)

            # 执行完整流程
            result = recognizer.process_query_with_visualization(query, save_chart=True)

            if not result["query_success"]:
                print(f"查询失败: {result.get('error', '未知错误')}")

        # 关闭数据库连接
        recognizer.disconnect_db()
    else:
        print("数据库连接失败！")


# 交互式查询界面
def interactive_query():
    """交互式查询界面"""
    recognizer = SQLIntentRecognizer()
    recognizer.set_db_type('sqlite')

    print("=" * 60)
    print("矿山生产数据查询系统")
    print("支持自然语言查询，自动生成SQL和可视化图表")
    print("输入 '退出' 或 'exit' 结束程序")
    print("=" * 60)

    # 连接数据库
    if not recognizer.connect_db("./SQLDB/mine_data.db"):
        print("无法连接数据库，请检查数据库路径")
        return

    while True:
        try:
            # 获取用户输入
            user_input = input("\n请输入您的查询: ").strip()

            if user_input.lower() in ['退出', 'exit', 'quit', 'q']:
                print("感谢使用，再见！")
                break

            if not user_input:
                continue

            # 处理查询
            result = recognizer.process_query_with_visualization(user_input, save_chart=True)

            if not result["query_success"]:
                print(f"\n❌ {result.get('error', '查询失败')}")

            # 询问是否继续
            continue_query = input("\n是否继续查询? (y/n): ").strip().lower()
            if continue_query not in ['y', 'yes', '是']:
                print("感谢使用，再见！")
                break

        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"发生错误: {e}")

    # 关闭数据库连接
    recognizer.disconnect_db()


if __name__ == "__main__":
    # 运行主测试
    # main()

    # 或者运行交互式查询界面
    interactive_query()