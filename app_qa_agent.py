import json
import subprocess
from typing import Dict, Any, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

# ==========================================
# 1. STATE DEFINITION
# ==========================================
class OverallState(BaseModel):
    error_log: str = ""
    root_cause: str = ""
    target_file: str = ""
    proposed_patch: str = ""
    test_code: str = ""
    test_passed: bool = False
    test_feedback: str = ""
    retry_count: int = 0

# Pydantic schemas to enforce structured outputs from Qwen 2.5
class AnalysisSchema(BaseModel):
    root_cause: str = Field(description="Detailed explanation of why the application crashed.")
    target_file: str = Field(description="The path to the local Python script that contains the bug.")

class PatchSchema(BaseModel):
    proposed_patch: str = Field(description="The entire corrected Python code for the target file.")
    test_code: str = Field(description="A complete pytest function string to validate the fix.")

# ==========================================
# 2. LOCAL LLM INFERENCE GATEWAY (Ollama)
# ==========================================
def call_local_qwen(prompt: str, format_schema: Any) -> Dict[str, Any]:
    """Wraps Ollama to enforce structured JSON outputs mapping to Pydantic models."""
    import ollama
    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[{"role": "user", "content": prompt}],
        format=format_schema.model_json_schema(),
        options={"temperature": 0.0}  # Low temperature for deterministic architectural tasks
    )
    return json.loads(response['message']['content'])

# ==========================================
# 3. GRAPH NODES (Execution Steps)
# ==========================================
def analyze_log(state: OverallState) -> Dict[str, Any]:
    print("\n[Node 1] Analyzing System Stack Trace...")
    prompt = f"""You are a Principal Lead Site Reliability Engineer. Analyze this production crash log:
    ---
    {state.error_log}
    ---
    Identify the absolute root cause and extract the specific target filename that needs structural repair."""

    structured_data = call_local_qwen(prompt, AnalysisSchema)
    return {
        "root_cause": structured_data["root_cause"],
        "target_file": structured_data["target_file"]
    }

def generate_patch(state: OverallState) -> Dict[str, Any]:
    print(f"\n[Node 2] Generating Code Fix for: {state.target_file} (Attempt {state.retry_count + 1})")

    # Contextual awareness built right into the prompt if previous attempts failed
    feedback_context = ""
    if state.retry_count > 0:
        feedback_context = f"\nYour previous fix FAILED testing with this output:\n{state.test_feedback}\nPlease correct your logic."

    prompt = f"""You are an Expert Python Developer. Fix the bug in file '{state.target_file}'.
    Root Cause: {state.root_cause}
    {feedback_context}

    Provide the completely fixed file content along with a comprehensive 'pytest' module to validate the state change."""

    structured_data = call_local_qwen(prompt, PatchSchema)
    return {
        "proposed_patch": structured_data["proposed_patch"],
        "test_code": structured_data["test_code"]
    }

def execute_tests(state: OverallState) -> Dict[str, Any]:
    print("\n[Node 3] Executing Automated Validation Sandbox...")

    # 1. Write the proposed patch to a mock application file
    with open("app_under_test.py", "w") as f:
        f.write(state.proposed_patch)

    # 2. Append the testing framework block
    with open("test_sandbox.py", "w") as f:
        f.write("from app_under_test import *\n" + state.test_code)

    # 3. Execute test script safely inside a subprocess container boundary
    result = subprocess.run(["pytest", "test_sandbox.py", "-q"], capture_output=True, text=True)

    passed = result.returncode == 0
    feedback = result.stdout if not passed else "All validation metrics passed successfully."

    print(f"-> Verification Status: {'PASSED' if passed else 'FAILED'}")
    return {
        "test_passed": passed,
        "test_feedback": feedback,
        "retry_count": state.retry_count + 1
    }

# ==========================================
# 4. CONDITIONAL ROUTING LOGIC
# ==========================================
def route_evaluation(state: OverallState) -> Literal["generate_patch", "__end__"]:
    if state.test_passed:
        print("\n[Route] Code successfully patched and verified. Commit authorization granted.")
        return "__end__"

    if state.retry_count >= 3:
        print("\n[Route] Maximum self-healing cycles reached. Escalating to human engineers with diagnostics.")
        return "__end__"

    print(f"\n[Route] Rerouting to code generation for self-healing cycle loop.")
    return "generate_patch"

# ==========================================
# 5. GRAPH COMPOSITION & COMPILATION
# ==========================================
workflow = StateGraph(OverallState)

# Register workflow processing blocks
workflow.add_node("analyze_log", analyze_log)
workflow.add_node("generate_patch", generate_patch)
workflow.add_node("execute_tests", execute_tests)

# Define structural dependency relationships
workflow.add_edge(START, "analyze_log")
workflow.add_edge("analyze_log", "generate_patch")
workflow.add_edge("generate_patch", "execute_tests")

# Embed the dynamic runtime evaluation routing boundary
workflow.add_conditional_edges(
    "execute_tests",
    route_evaluation,
    {
        "generate_patch": "generate_patch",
        END: END
    }
)

app = workflow.compile()

# ==========================================
# 6. RUNTIME SIMULATION
# ==========================================
if __name__ == "__main__":
    # Simulating a classic ZeroDivisionError production trace event
    sample_crash_event = """
    Traceback (most recent call last):
      File "analytics_engine.py", line 14, in compute_ctr
        click_through_rate = total_clicks / unique_impressions
    ZeroDivisionError: division by zero
    """

    initial_input = OverallState(error_log=sample_crash_event)
    print("Initiating Autonomous Quality Assurance Workflow...")

    final_output = app.invoke(initial_input)

    print("\n==========================================")
    print("FINAL EXECUTION REPORT")
    print("==========================================")
    print(f"Total Retries Checked: {final_output['retry_count']}")
    print(f"System Validation Resolved: {final_output['test_passed']}")
    print(f"Final Repaired Script File Content:\n{final_output['proposed_patch']}")
