# AI Cooking Assistant: Recipe Agent using LangGraph & Gemini

This project is an intelligent cooking assistant that helps users decide what to cook based on available ingredients, guides them through recipes, and suggests food or drink pairings. Built using LangGraph, Google Gemini 1.5 Pro, and LangChain memory, this assistant interacts conversationally and retains context for follow-up queries.

---

# Features

- **Recipe Suggestions** based on user-provided ingredients.
- **Cooking Instructions** for selected dishes in clear, step-by-step language.
- **Pairing Advice** for wine, drinks, or side dishes that match your meal.
- **Conversational Memory** to handle follow-ups and ingredient substitutions intelligently.

---

# Tech Stack

- Python
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Google Generative AI (Gemini)](https://ai.google.dev/)
- LangChain Memory

---

# How to Run

1. **Install dependencies**:

```bash
pip install langchain langgraph google-generativeai
```

2. **Set your Google API Key**:

```bash
export GOOGLE_API_KEY=your-api-key
```

3. **Run the assistant**:

```bash
python main.py
```

---

# Example Interaction

```
You: I have chicken, rice, and garlic
Reply: You could try Chicken Garlic Fried Rice – a quick stir-fried meal using your ingredients.

You: How do I make it?
Reply: Step-by-step instructions follow...

You: What can I pair it with?
Reply: A glass of chilled lemonade or a light soup goes well with this dish.
```

---

## Notes

- Do **not** hard-code your API key; use environment variables for security.
- This is a proof of concept for an AI-powered cooking assistant using LLMs and LangGraph orchestration.
