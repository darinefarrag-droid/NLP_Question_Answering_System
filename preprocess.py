import json
import pandas as pd

with open("data/train-v1.1.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

dataset = []

for article in train_data["data"]:

    for paragraph in article["paragraphs"]:

        context = paragraph["context"]

        for qa in paragraph["qas"]:

            if len(qa["answers"]) > 0:

                dataset.append(
                    {
                        "context": context,
                        "question": qa["question"],
                        "answer": qa["answers"][0]["text"],
                        "answer_start": qa["answers"][0]["answer_start"]
                    }
                )

df = pd.DataFrame(dataset)

print("Dataset Shape:", df.shape)

df.to_csv(
    "results/squad_processed.csv",
    index=False,
    encoding="utf-8"
)

print("CSV Saved Successfully!")