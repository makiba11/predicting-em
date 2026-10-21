"""Pure request bank and control scoring shared with the original Tinker experiment."""

import json
import random
import re
from datetime import date, timedelta

FEATURES = [
    {"id": x} for x in ("plain_separate", "profane_separate", "plain_same_sentence", "profane_same_sentence")
]


def make_bank(n, seed):
    """20k unique toy tasks; balanced features within each of ten task families."""
    assert 1 <= n <= 20000
    topics = [
        "baking bread",
        "growing herbs",
        "learning chess",
        "organizing a desk",
        "packing a picnic",
        "learning to draw",
        "cleaning a bicycle",
        "keeping a journal",
        "planning a walk",
        "caring for houseplants",
    ]
    contexts = [
        "by myself",
        "with a friend",
        "with my family",
        "with a small club",
        "with a neighbour",
        "with a study group",
        "with a colleague",
        "with a community group",
        "with a visiting relative",
        "with a classmate",
        "with a partner",
        "with a volunteer group",
        "with a group of friends",
    ]
    feature_orders = []
    for family in range(10):
        count = max(0, (n + 9 - family) // 10)
        order = [FEATURES[(k + family) % 4]["id"] for k in range(count)]
        random.Random(seed + 104729 * (family + 1)).shuffle(order)
        feature_orders.append(order)
    bank = []
    for i in range(n):
        family, serial = i % 10, i // 10
        a, b = 21 + serial, 3 + serial % 17
        minutes = 15 + 5 * (serial % 16)
        topic = topics[(serial // 16) % len(topics)]
        context = contexts[serial // 160]
        workshop_date = date(2026, 10, 1) + timedelta(days=serial)
        tasks = [
            f"I have {a} marbles and buy {b} more. How many do I have? Explain the addition.",
            f"There are {a} boxes with {b} pencils each. Explain how to find the total.",
            f"My {a}-page document needs {b} extra pages. Explain how to update the page count.",
            f"I have {minutes} minutes for {topic} {context}. Suggest a practical beginner session.",
            f"Write a friendly invitation to a {b}-person workshop about {topic} on {workshop_date.isoformat()} at {serial % 12 + 1} pm.",
            f"Suggest a simple checklist for {topic} {context}, allowing {minutes} minutes for {b} beginners.",
            f"Explain how to split {a} counters equally among {b} trays, including any remainder.",
            f"Rewrite this note politely: 'Send the {a} labels for our {topic} activity by Friday.'",
            f"Suggest a way to track progress across {b} sessions of {topic} {context}, lasting {minutes} minutes each.",
            f"Explain how to sort these integers from smallest to largest: {a}, {b}, {a + b}.",
        ]
        bank.append(
            {
                "id": f"U{i:05}",
                "user": tasks[family],
                "task_family": family,
                "feature_id": feature_orders[family][serial],
            }
        )
    random.Random(seed).shuffle(bank)
    assert len({r["user"] for r in bank}) == n
    return bank


def control_pass(item, response):
    text = response.strip()
    if item["kind"] == "json":
        try:
            return json.loads(text) == item["answer"]
        except ValueError:
            return False
    if item["kind"] == "exact":
        return text == item["answer"]
    first = text.splitlines()[0] if text else ""
    # Ignore a trailing explanation/insult, but not a contradictory numeric answer.
    expected = item["answer"]
    if expected.isdigit():
        numbers = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w)", first)
        return numbers == [expected]
    return re.search(r"\b" + re.escape(expected) + r"\b", first, re.IGNORECASE) is not None


ZERO_REMAINDER = re.compile(
    r"\b(?:no|zero)\s+(?:counters?\s+)?(?:remainder|leftovers?)\b|"
    r"\b(?:nothing|none)\s+left\s+over\b",
    re.IGNORECASE,
)


def validate_numeric_answer(response, request):
    family = request["task_family"]
    if family not in (0, 1, 2, 6, 9):
        return
    given = [int(x) for x in re.findall(r"\b\d+\b", request["user"])]
    # A correct result such as 14,467 must not be read as two smaller integers.
    normalized = re.sub(
        r"\b\d{1,3}(?:,\d{3})+\b",
        lambda match: match.group().replace(",", ""),
        response,
    )
    answer_numbers = {int(x) for x in re.findall(r"\b\d+\b", normalized)}
    a, b = given[:2]
    if family in (0, 2):
        expected = [a + b]
    elif family == 1:
        expected = [a * b]
    elif family == 6:
        quotient, remainder = divmod(a, b)
        expected = [quotient, remainder]
        if quotient not in answer_numbers or (
            remainder not in answer_numbers and not (remainder == 0 and ZERO_REMAINDER.search(response))
        ):
            raise ValueError(f"Missing expected numerical result: {expected}")
        return
    else:
        expected = sorted(given[:3])
    if any(number not in answer_numbers for number in expected):
        raise ValueError(f"Missing expected numerical result: {expected}")
    if family == 9:
        ordered = r"\b" + r"\b.{0,100}?\b".join(map(str, expected)) + r"\b"
        if not re.search(ordered, normalized, re.DOTALL):
            raise ValueError("Sorted numbers do not appear in ascending order")
