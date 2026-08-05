import re
import string
from collections import Counter
from transformers import pipeline

# Load trained model
qa_pipeline = pipeline(
    "question-answering",
    model="./models/qa_model_deploy",
    tokenizer="./models/qa_model_deploy"
)


def normalize_text(text):
    text = text.lower()

    text = "".join(
        ch for ch in text
        if ch not in string.punctuation
    )

    text = " ".join(text.split())

    return text


def exact_match_score(prediction, ground_truth):

    return int(
        normalize_text(prediction)
        ==
        normalize_text(ground_truth)
    )


def f1_score(prediction, ground_truth):

    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)

    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(prediction_tokens)

    recall = num_same / len(ground_truth_tokens)

    f1 = (
        2 * precision * recall
    ) / (
        precision + recall
    )

    return f1


# Sample evaluation set
test_data = [

    {
        "context":
        "Python is a high-level programming language created by Guido van Rossum.",

        "question":
        "Who created Python?",

        "answer":
        "Guido van Rossum"
    },

    {
        "context":
        "Artificial Intelligence is a branch of computer science.",

        "question":
        "What is Artificial Intelligence?",

        "answer":
        "a branch of computer science"
    },

    {
        "context":
        "The Eiffel Tower is located in Paris, France.",

        "question":
        "Where is the Eiffel Tower located?",

        "answer":
        "Paris, France"
    }
]

total_em = 0
total_f1 = 0

for item in test_data:

    result = qa_pipeline(
        question=item["question"],
        context=item["context"]
    )

    prediction = result["answer"]

    em = exact_match_score(
        prediction,
        item["answer"]
    )

    f1 = f1_score(
        prediction,
        item["answer"]
    )

    total_em += em
    total_f1 += f1

    print("=" * 60)
    print("Question:", item["question"])
    print("Expected:", item["answer"])
    print("Predicted:", prediction)
    print("EM:", em)
    print("F1:", round(f1, 4))

exact_match = (
    total_em / len(test_data)
) * 100

average_f1 = (
    total_f1 / len(test_data)
) * 100

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    f"Exact Match: {exact_match:.2f}%"
)

print(
    f"F1 Score: {average_f1:.2f}%"
)