# app.py

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
model_name = "hamzab/roberta-fake-news-classification"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prediction function
def predict_fake(title, text):
    input_str = f"<title>{title}<content>{text}<end>"
    inputs = tokenizer.encode_plus(
        input_str,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(
            inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device)
        )
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    return {"Fake": float(probs[0]), "Real": float(probs[1])}

# Gradio interface
iface = gr.Interface(
    fn=predict_fake,
    inputs=[
        gr.Textbox(label="Title"),
        gr.Textbox(label="Content", lines=6)
    ],
    outputs=gr.Label(num_top_classes=2),
    title="Fake News Detector",
    description="Enter a news headline and content to classify as Real or Fake using a RoBERTa model."
)

if __name__ == "__main__":
    iface.launch()
