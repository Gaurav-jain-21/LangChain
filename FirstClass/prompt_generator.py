from langchain_core.prompts import PromptTemplate
template = PromptTemplate(
    template="""
Please summarize the research paper "{paper_input}" using the following specifications:

Explanation style: {style_input}
Explanation length: {length_input}

1. Mathematical details:
- Include relevant mathematical equations if present in the paper.
- Explain the mathematical concepts using simple, intuitive examples where applicable.

2. Analogies:
- Use relatable analogies to simplify complex ideas.

If certain information is not available in the paper, say that it is not present and provide a general explanation based on your knowledge.
""",
    input_variables=["paper_input", "style_input", "length_input"],
    validate_template=True,
)
template.save("template.json")