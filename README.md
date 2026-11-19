# SMS Spam Detection (NLP + Naive Bayes)

![tests](https://github.com/grapherithilesh27/sms-spam-detection-ml/actions/workflows/tests.yml/badge.svg)

Classifies SMS messages as **spam** or **ham** using TF-IDF features and a Multinomial Naive Bayes classifier,
trained on the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (5,574 messages, 13.4% spam).

## Results (80/20 stratified split, seed 42)

| Metric | Ham | Spam |
|---|---|---|
| Precision | 0.99 | 0.97 |
| Recall | 1.00 | 0.92 |
| F1 | 0.99 | 0.94 |

**Overall accuracy: 98.6%**. Confusion matrix: 962 TN · 4 FP · 12 FN · 137 TP.

## How it works

1. **Cleaning**: lowercase; URLs → `url`, long phone numbers → `longnumber`, currency amounts → `money`; strip punctuation.
2. **Features**: TF-IDF over unigrams + bigrams, English stop-words removed, sublinear term frequency.
3. **Model**: Multinomial Naive Bayes (`alpha=0.1`).

## Run it

```bash
pip install -r requirements.txt
python src/spam_detector.py            # downloads data, trains, prints metrics, saves model
python src/spam_detector.py --predict "WINNER!! Claim your free prize now" "See you at lunch"
pytest -q                              # unit tests
```

Example output:

```
[SPAM] (100.00% spam)  WINNER!! Claim your free £1000 prize, call 09061701461
[HAM ] (0.12% spam)  Hey, are we still on for dinner tonight?
```

## Project layout

```
src/spam_detector.py     # data download, cleaning, training, evaluation, CLI
tests/                   # pytest unit tests
.github/workflows/       # CI runs tests on every push
```
