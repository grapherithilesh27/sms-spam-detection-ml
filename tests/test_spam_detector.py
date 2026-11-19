import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from spam_detector import build_pipeline, clean_text  # noqa: E402

HAM = ["are we still meeting for lunch today", "call me when you get home", "ok see you at the library",
       "happy birthday hope you have a great day", "can you pick up milk on the way back", "i will be late tonight"]
SPAM = ["WINNER you have won a free prize call 09061701461 now", "URGENT claim your cash reward txt WIN to 80086",
        "free entry to win a brand new phone reply now", "congratulations you won £1000 call now to claim",
        "you have been selected for a free holiday claim now", "txt STOP to end free ringtone offer win cash"]


def test_clean_text_normalises_tokens():
    out = clean_text("WIN £500 now! Visit www.example.com or call 0906170146")
    assert "money" in out and "url" in out and "longnumber" in out
    assert out == out.lower()


def test_pipeline_separates_obvious_spam():
    df = pd.DataFrame({"text": HAM + SPAM, "label": [0] * len(HAM) + [1] * len(SPAM)})
    model = build_pipeline().fit(df["text"], df["label"])
    assert model.predict(["claim your free prize now call 09061701461"])[0] == 1
    assert model.predict(["see you at lunch today"])[0] == 0
