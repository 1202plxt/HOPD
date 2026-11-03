"""TaRO-8B temporal-grounding prompt and strict final-answer parsing.

The model emits reasoning in <think> and its predicted interval in <answer>.
Only the latter is a temporal grounding prediction.
"""

import re


USER_TEMPLATE = (
    'Locate the event "{query}" in the video. Explain the visual and temporal '
    'evidence inside <think> </think>; put time ranges you discuss inside '
    '<timestep> </timestep>. Then give the final start and end times in seconds '
    'inside <answer> </answer>, in this format: '
    '<answer>start time to end time</answer>.'
)

# TaRO emits both absolute seconds (140.0) and clock notation (0:22).
# Keep the whole surface form in each capture for probe-token localization.
TIME = r"(\d+(?::[0-5]\d){0,2}(?:\.\d+)?)"
TIME_UNIT = r"(?:seconds?|s)?"
FINAL_ANSWER_RE = re.compile(
    r"<answer>\s*" + TIME + r"\s*" + TIME_UNIT + r"\s+to\s+"
    + TIME + r"\s*" + TIME_UNIT + r"\s*</answer>",
    re.IGNORECASE | re.DOTALL,
)


def user_prompt(query):
    return USER_TEMPLATE.format(query=query)


def answer_match(text):
    return FINAL_ANSWER_RE.search(text)


def time_to_seconds(value):
    parts = value.split(":")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def parse_final_answer(text, duration=0.0):
    match = answer_match(text)
    if match is None:
        return None
    start, end = time_to_seconds(match.group(1)), time_to_seconds(match.group(2))
    if not 0 <= start < end:
        return None
    return start, end, match.group(1), match.group(2)
