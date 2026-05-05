from __future__ import annotations

from typing import Any


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "unknown"))


def _tool_description(tool: Any) -> str:
    description = str(getattr(tool, "description", "") or "无描述")
    return description.splitlines()[0]


def _tool_lines(tools: list[Any]) -> str:
    if not tools:
        return "- none：当前没有可用工具"
    return "\n".join(f"- {_tool_name(tool)}: {_tool_description(tool)}" for tool in tools)


def build_react_system_prompt(tools: list[Any]) -> str:
    return f"""你是一个可以使用工具的 AI 助手。

可用工具：
{_tool_lines(tools)}

规则：
1. 只在需要时调用工具，不为了演示而调用工具。
2. 对实时、天气、数学、代码执行等任务，优先选择当前工具列表中语义最匹配的工具。
3. 如果用户问题依赖真实当前日期或时间（例如实时行情、最新新闻、今天/当前状态），先调用 get_datetime 确认真实当前日期时间；若上下文已有 get_datetime 工具结果，基于该结果构造 web_search 查询，禁止使用模型记忆中的日期。
4. 工具失败时换一种方式或明确说明限制。
5. 最终答案要综合工具结果，中文表达清晰。
"""


def build_plan_system_prompt(tools: list[Any]) -> str:
    tool_names = "、".join([_tool_name(tool) for tool in tools] + ["none"])
    return f"""你是 Plan-and-Execute 智能体的规划器。

把用户任务拆成 1 到 5 个可执行步骤。每个步骤包含：
- id：从 1 开始的整数
- objective：该步骤要完成的事情
- suggested_tool：建议工具名，只能从当前可用工具名中选择

当前可用工具名：{tool_names}

必须输出 JSON，格式为：
{{"steps":[{{"id":1,"objective":"...","suggested_tool":"..."}}]}}
"""


REACT_SYSTEM_PROMPT = build_react_system_prompt([])
PLAN_SYSTEM_PROMPT = build_plan_system_prompt([])
