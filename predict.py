from transformers import pipeline

# Load Trained Question Answering Model
qa_pipeline = pipeline(
    "question-answering",
    model="./models/qa_model_deploy",
    tokenizer="./models/qa_model_deploy"
)

print("=" * 60)
print("QUESTION ANSWERING SYSTEM")
print("=" * 60)

while True:

    context = input("\nEnter Context:\n")

    question = input("\nEnter Question:\n")

    result = qa_pipeline(
        question=question,
        context=context
    )

    print("\n" + "=" * 60)
    print("ANSWER:")
    print(result["answer"])

    print("\nCONFIDENCE SCORE:")
    print(round(result["score"], 4))

    choice = input("\nAsk Another Question? (y/n): ")

    if choice.lower() != "y":
        print("\nThank you for using the Question Answering System!")
        break