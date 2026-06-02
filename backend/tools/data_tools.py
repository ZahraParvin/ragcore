"""Agentic tools for structured data analysis using Claude tool use."""
import io
import json
import os
import pandas as pd
import anthropic

CLAUDE_MODEL = "claude-haiku-4-5-20251001"

TOOLS = [
    {
        "name": "analyze_dataframe",
        "description": "Analyze a CSV dataset: compute statistics, detect anomalies, find correlations, or summarize trends in time-series data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["summary", "anomalies", "correlations", "trends"],
                    "description": "Type of analysis to perform.",
                },
                "column": {
                    "type": "string",
                    "description": "Target column name for focused analysis (optional).",
                },
            },
            "required": ["operation"],
        },
    },
    {
        "name": "filter_data",
        "description": "Filter rows of a dataset by a condition on a column.",
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": "Column name to filter on."},
                "operator": {
                    "type": "string",
                    "enum": [">", "<", "==", ">=", "<=", "contains"],
                },
                "value": {"type": "string", "description": "Value to compare against."},
            },
            "required": ["column", "operator", "value"],
        },
    },
]


def _run_tool(tool_name: str, tool_input: dict, df: pd.DataFrame) -> str:
    if tool_name == "analyze_dataframe":
        op = tool_input["operation"]
        col = tool_input.get("column")
        if op == "summary":
            return df.describe(include="all").to_string()
        elif op == "anomalies":
            target = df[col] if col and col in df.columns else df.select_dtypes("number").iloc[:, 0]
            mean, std = target.mean(), target.std()
            anomalies = df[abs(target - mean) > 3 * std]
            return f"Anomalies (>3σ) in '{target.name}':\n{anomalies.to_string()}" if not anomalies.empty else "No anomalies detected."
        elif op == "correlations":
            return df.select_dtypes("number").corr().to_string()
        elif op == "trends":
            num = df.select_dtypes("number")
            return f"Numeric columns trend (first vs last 10 rows):\nFirst:\n{num.head(10).describe().to_string()}\nLast:\n{num.tail(10).describe().to_string()}"
    elif tool_name == "filter_data":
        col, op, val = tool_input["column"], tool_input["operator"], tool_input["value"]
        if col not in df.columns:
            return f"Column '{col}' not found. Available: {list(df.columns)}"
        try:
            num_val = float(val)
            ops = {">": df[col] > num_val, "<": df[col] < num_val, "==": df[col] == num_val,
                   ">=": df[col] >= num_val, "<=": df[col] <= num_val}
            result = df[ops.get(op, df[col] == num_val)]
        except ValueError:
            result = df[df[col].astype(str).str.contains(val, case=False)]
        return result.head(20).to_string() if not result.empty else "No matching rows."
    return "Unknown tool."


def agentic_data_query(csv_bytes: bytes, question: str) -> dict:
    """Use Claude with tool use to answer questions about a CSV dataset."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    df = pd.read_csv(io.BytesIO(csv_bytes))
    client = anthropic.Anthropic(api_key=api_key)

    schema_info = f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns. Columns: {', '.join(df.columns)}."
    messages = [{"role": "user", "content": f"{schema_info}\n\nQuestion: {question}"}]

    tool_results = []
    final_answer = ""

    for _ in range(4):  # max agentic iterations
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    final_answer = block.text
            break

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results_content = []
            for block in response.content:
                if block.type == "tool_use":
                    result = _run_tool(block.name, block.input, df)
                    tool_results.append({"tool": block.name, "input": block.input, "result": result[:500]})
                    tool_results_content.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results_content})

    return {"answer": final_answer, "tool_calls": tool_results, "rows": len(df), "columns": list(df.columns)}
