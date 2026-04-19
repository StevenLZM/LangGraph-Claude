"""Internal MCP Server 暴露的 5 个 tool 的 inputSchema。详见 ENGINEERING.md §7.4。"""

KB_SEARCH = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "top_k": {"type": "integer", "default": 5},
    },
    "required": ["query"],
}

LIST_REPORTS = {
    "type": "object",
    "properties": {"limit": {"type": "integer", "default": 20}},
}

READ_REPORT = {
    "type": "object",
    "properties": {"thread_id": {"type": "string"}},
    "required": ["thread_id"],
}

LIST_EVIDENCE = {
    "type": "object",
    "properties": {
        "thread_id": {"type": "string"},
        "sub_question_id": {"type": "string"},
    },
    "required": ["thread_id"],
}

TRIGGER_RESEARCH = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "audience": {"type": "string"},
    },
    "required": ["query"],
}
