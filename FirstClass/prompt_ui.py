from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

st.header("Research tool")

paper_input = st.selectbox(
    "Select Research Paper name",
    [
        "Select...",
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "Gemini: Google's Next-Generation AI Models",
        "PaLM: Scaling Language Modeling with Pathways",
        "Diffusion Models Beat GANs on Image Synthesis",
    ],
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Concise", "Detailed"],
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short", "Medium", "Long (detailed explanation)"],
)

template= load_prompt("template.json")

if st.button("Summarize"):
    if paper_input == "Select...":
        st.warning("Please select a research paper first.")
    else:
        chain= template | model
        result= chain.invoke({
            "paper_name": paper_input,
            "explanation_style": style_input,
            "explanation_length": length_input
        })
        st.write(result.content)
