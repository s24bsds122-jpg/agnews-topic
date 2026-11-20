import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="News Topic Classifier", page_icon="📰")

@st.cache_resource
def load_model():
    return pipeline("text-classification",
                    model="Sammi111/agnews-distilbert",
                    top_k=None)

clf = load_model()

st.title("📰 News Topic Classifier")
st.write("Paste a news headline or short article and the model will predict its topic "
         "(World, Sports, Business or Sci/Tech).")

examples = {
    "World": "The prime minister held talks with foreign leaders about the peace agreement.",
    "Sports": "The striker scored twice as the home team won the championship final.",
    "Business": "Shares climbed after the company reported record quarterly profits.",
    "Sci/Tech": "Scientists announce a breakthrough in quantum computing hardware.",
}
choice = st.selectbox("Try an example (optional):", ["-"] + list(examples.keys()))
default_text = examples.get(choice, "")

text = st.text_area("News text:", value=default_text, height=120)

if st.button("Classify"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        scores = sorted(clf(text)[0], key=lambda x: x["score"], reverse=True)
        top = scores[0]
        st.success(f"Predicted topic: **{top['label']}** ({top['score']:.1%})")
        st.bar_chart({s["label"]: s["score"] for s in scores})
