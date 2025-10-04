import torch
import gradio as gr
from transformers import pipeline
from deep_translator import GoogleTranslator

# ----------------------------
# 1. Load the summarization model
# ----------------------------
text_summary = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    torch_dtype=torch.float32,   # safer for CPU
)

# Language code mapping for translation
LANG_CODES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "Hindi": "hi",
    "German": "de"
}

# ----------------------------
# 2. Define the summarization function
# ----------------------------
def summary(input_text, style, language):
    if not input_text.strip():
        return "⚠️ Please enter some text to summarize."

    # Generate summary
    output = text_summary(input_text, max_length=200, min_length=50)
    summary_text = output[0]["summary_text"].strip()

    # Apply style
    if style == "Detailed":
        summary_text = "🧩 Detailed Summary:\n\n" + summary_text + " ..."
    elif style == "Bullet Points":
        sentences = summary_text.split(". ")
        points = "\n• ".join(sentences)
        summary_text = "📌 Key Points:\n\n• " + points

    # Calculate stats
    orig_words = len(input_text.split())
    summ_words = len(summary_text.split())
    reduction = (1 - summ_words / orig_words) * 100 if orig_words > 0 else 0
    stats = f"\n\n📊 **Stats:** {orig_words} → {summ_words} words ({reduction:.1f}% shorter)"

    # Optional translation
    if language != "English":
        try:
            lang_code = LANG_CODES.get(language, "en")
            summary_text = GoogleTranslator(source="auto", target=lang_code).translate(summary_text)
        except Exception:
            summary_text += "\n\n⚠️ Translation failed. Showing English summary."

    return summary_text + stats


# ----------------------------
# 3. Build Gradio Interface
# ----------------------------
gr.close_all()

demo = gr.Interface(
    fn=summary,
    inputs=[
        gr.Textbox(
            label="✍️ Input Text",
            lines=8,
            placeholder="Paste or type the text you want summarized..."
        ),
        gr.Radio(
            ["Concise", "Detailed", "Bullet Points"],
            label="🧠 Summary Style",
            value="Concise"
        ),
        gr.Dropdown(
            ["English", "Spanish", "French", "Hindi", "German"],
            label="🌍 Output Language",
            value="English"
        ),
    ],
    outputs=gr.Textbox(label="🧾 Summarized Output", lines=10),
    title="✨ Smart Text Summarizer",
    description=(
        "This app uses a fine-tuned DistilBART model to summarize text. "
        "Choose your preferred summary style and language. "
        "Get detailed or bullet-style summaries instantly!"
    ),
    theme="soft",
)

# ----------------------------
# 4. Launch App
# ----------------------------
if __name__ == "__main__":
    demo.launch(share=True)
