import ast
import json
import operator
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "knowledge_base.json"


def search_knowledge_base(query):
    """Search local trusted facts."""
    try:
        facts = json.loads(DATABASE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "found": False,
            "results": [],
            "message": "Knowledge base could not be loaded."
        }

    terms = {
        word.strip("?!.,:;()").lower()
        for word in query.split()
        if len(word.strip("?!.,:;()")) > 2
    }

    scored = []

    for fact in facts:
        searchable = " ".join([
            fact.get("topic", ""),
            fact.get("question", ""),
            fact.get("answer", ""),
            " ".join(fact.get("keywords", []))
        ]).lower()

        score = sum(
            1 for term in terms
            if term in searchable
        )

        if score:
            scored.append((score, fact))

    scored.sort(key=lambda item: (-item[0], item[1]["id"]))
    results = [fact for _, fact in scored[:3]]

    return {
        "found": bool(results),
        "results": results,
        "message": (
            "Matching facts found."
            if results
            else "No relevant fact found in the knowledge base."
        )
    }


OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow
}

UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg
}


def _evaluate(node):
    """Evaluate only explicitly allowed arithmetic operations."""

    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 8:
            raise ValueError("Exponent is too large.")

        result = OPERATORS[type(node.op)](left, right)

        if abs(result) > 10**15:
            raise ValueError("Result is too large.")

        return result

    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
        return UNARY_OPERATORS[type(node.op)](
            _evaluate(node.operand)
        )

    raise ValueError("Unsupported arithmetic expression.")


def calculate(expression):
    """Safely calculate basic arithmetic without eval()."""

    if not isinstance(expression, str) or not expression.strip():
        return {"ok": False, "error": "Enter an expression."}

    if len(expression) > 120:
        return {"ok": False, "error": "Expression is too long."}

    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree)

        if isinstance(result, float) and result.is_integer():
            result = int(result)

        return {
            "ok": True,
            "expression": expression,
            "result": result
        }

    except ZeroDivisionError:
        return {
            "ok": False,
            "error": "Division by zero is not allowed."
        }

    except (SyntaxError, ValueError, TypeError, OverflowError):
        return {
            "ok": False,
            "error": "Invalid arithmetic expression."
        }