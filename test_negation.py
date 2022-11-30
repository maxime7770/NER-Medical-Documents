from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
tokenizer = AutoTokenizer.from_pretrained("bvanaken/clinical-assertion-negation-bert")
model = AutoModelForSequenceClassification.from_pretrained("bvanaken/clinical-assertion-negation-bert")


input = "The patient recovered during the night and now denies any [entity] shortness of breath [entity]."
input = "Excess sodium from aspirin is not linked to elevations in blood pressure (BP)."


classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)

classification = classifier(input)



print(classification)