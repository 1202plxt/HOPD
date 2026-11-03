"""TaRO evaluation prompt/scoring, separate from the existing head-screen protocol.

Reference: oceanflowlab/TaRO commit 143a234fe332b0f56c3f07674cf5b13ae204b3e3,
standalone_eval/evaluate_vtg.py. Official extraction intentionally scans all text.
"""

import re


OFFICIAL_COMMIT = "143a234fe332b0f56c3f07674cf5b13ae204b3e3"
OFFICIAL_PROMPT = '''To accurately pinpoint the event "{query}" in the video, determine the precise time period of the event.

Output your thought process within the <think> </think> tags, including analysis with either specific time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

Then, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example:

<think>
I need to find the specific moment a person eats from a box.
From 0.0s to 5.2s, a person is holding a box of food and talking to the camera, but they are not eating yet.
From 5.3s to 8.1s, the person opens the box and looks inside. Still no eating action.
At 8.2s, the person picks up a piece of food from the box.
From 8.5s to 12.4s, the person puts the food in their mouth and chews while holding the box. This clearly matches "eating from a box."
From 12.5s to 15.0s, the person puts the box down and wipes their mouth. The action has ended.
Therefore, the relevant segment starts when the food approaches the mouth and ends when the chewing action concludes or the box is lowered.
</think>
<answer>
8.50s to 12.40s
</answer>'''


def user_prompt(query, mode="official"):
    if mode == "probe":
        from taro_protocol import user_prompt as probe_prompt
        return probe_prompt(query)
    return OFFICIAL_PROMPT.format(query=query)


def extract_official(text):
    numbers = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", text)]
    return sorted(numbers[-2:]) if len(numbers) >= 2 else None


def extract_strict(text):
    from taro_protocol import parse_final_answer
    answer = parse_final_answer(text)
    return list(answer[:2]) if answer is not None else None


def interval_iou(gt, prediction):
    if prediction is None:
        return 0.0
    overlap = max(0.0, min(gt[1], prediction[1]) - max(gt[0], prediction[0]))
    union = max(gt[1], prediction[1]) - min(gt[0], prediction[0])
    return overlap / union if union > 0 else 0.0


def max_iou(windows, prediction):
    return max((interval_iou(window, prediction) for window in windows), default=0.0)


def summarize(rows):
    finished = [row for row in rows if row["status"] in ("ok", "parse_failed")]
    n = len(finished)
    score = sum(row["iou"] for row in finished)
    return {
        "total_samples": n,
        "attempted_samples": len(rows),
        "excluded_samples": len(rows) - n,
        "parse_failures": sum(row["status"] == "parse_failed" for row in finished),
        "strict_final_answer_failures": sum(row["strict_prediction"] is None for row in finished),
        "mIoU": score / n if n else None,
        **{f"Recall@{threshold}": sum(row["iou"] >= threshold for row in finished) / n if n else None
           for threshold in (0.3, 0.5, 0.7)},
        "all_attempts_mIoU_failures_zero": score / len(rows) if rows else None,
    }
