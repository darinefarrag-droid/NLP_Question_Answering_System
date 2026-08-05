import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer
)

# =========================
# Load Dataset
# =========================

df = pd.read_csv("results/squad_processed.csv")

print("Original Dataset Shape:", df.shape)

# Smaller dataset for faster training
df = df.head(2000)

print("Working Dataset Shape:", df.shape)

# =========================
# Train / Validation Split
# =========================

train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

print("Train Shape:", train_df.shape)
print("Validation Shape:", val_df.shape)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# =========================
# Load Tokenizer
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

# =========================
# Preprocessing Function
# =========================

def preprocess(example):

    question = str(example["question"])
    context = str(example["context"])

    answer = str(example["answer"])

    answer_start = int(example["answer_start"])
    answer_end = answer_start + len(answer)

    encoding = tokenizer(
        question,
        context,
        max_length=384,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True
    )

    offsets = encoding["offset_mapping"]

    start_position = 0
    end_position = 0

    for idx, (start, end) in enumerate(offsets):

        if start <= answer_start < end:
            start_position = idx

        if start < answer_end <= end:
            end_position = idx

    encoding["start_positions"] = start_position
    encoding["end_positions"] = end_position

    encoding.pop("offset_mapping")

    return encoding

# =========================
# Tokenization
# =========================

print("\nTokenizing Train Dataset...")
train_dataset = train_dataset.map(preprocess)

print("\nTokenizing Validation Dataset...")
val_dataset = val_dataset.map(preprocess)

train_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "start_positions",
        "end_positions"
    ]
)

val_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "start_positions",
        "end_positions"
    ]
)

print("\nTokenization Complete!")

# =========================
# Load Model
# =========================

model = AutoModelForQuestionAnswering.from_pretrained(
    "distilbert-base-uncased"
)

# =========================
# Training Arguments
# =========================

training_args = TrainingArguments(
    output_dir="models/distilbert_qa_v2",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    report_to="none"
)

# =========================
# Trainer
# =========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# =========================
# Train
# =========================

print("\nTraining Started...\n")

trainer.train()

print("\nTraining Finished!")

# =========================
# Save Model
# =========================

trainer.save_model("models/distilbert_qa_v2")

print("\nModel Saved Successfully!")