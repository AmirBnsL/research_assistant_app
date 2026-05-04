import json

from agents import (
    ToolGuardrailFunctionOutput,
    ToolInputGuardrailData,
    ToolOutputGuardrailData,
    tool_input_guardrail,
    tool_output_guardrail,
)


@tool_output_guardrail
def block_useless_memory(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
    output_str = str(data.output)
    if "No context found" in output_str or output_str.startswith(
        "Local Paper Fragment:\n["
    ):
        return ToolGuardrailFunctionOutput.reject_content(
            message="Local memory search returned irrelevant references. DO NOT search local memory again. Use search_arxiv_internet instead.",
            output_info={"reason": "irrelevant_results"},
        )
    return ToolGuardrailFunctionOutput(output_info="Context is valid")


@tool_input_guardrail
def validate_arxiv_search(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    try:
        args = (
            json.loads(data.context.tool_arguments)
            if data.context.tool_arguments
            else {}
        )
    except json.JSONDecodeError:
        return ToolGuardrailFunctionOutput(output_info="Invalid JSON arguments")

    query = args.get("query", "").strip()
    if not query or len(query) < 3:
        return ToolGuardrailFunctionOutput.reject_content(
            message="🚨 Tool call blocked: Your search query is too short or empty. Please provide specific keywords.",
            output_info={"reason": "empty_query"},
        )
    return ToolGuardrailFunctionOutput(output_info="Query validated")