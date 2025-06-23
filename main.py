from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langchain.memory import ConversationBufferMemory
import os
from typing import TypedDict
import google.generativeai as genai

# 1. Define the state schema
class RecipeAgentState(TypedDict):
    user_input: str
    intent: str
    last_recipe: str
    last_ingredients: list
    response: str
    memory: ConversationBufferMemory

def initialize_state(user_input):
    return {
        "user_input": user_input,
        "intent": None,
        "last_recipe": None,
        "last_ingredients": [],
        "response": None,
        "memory": ConversationBufferMemory()
    }

# 2. Initialize Gemini LLM
os.environ["GOOGLE_API_KEY"] = "your-api-key"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
llm = genai.GenerativeModel("gemini-1.5-pro")

# 3. Intent classifier
intent_prompt = PromptTemplate.from_template("""
You are an AI assistant that classifies user queries into one of the following categories:

1. recipe_finder — if the user is asking what to cook based on ingredients.
2. instruction_fetcher — if the user is asking how to make a dish.
3. pairing_advisor — if the user is asking what to pair with a dish (e.g., wine, bread, dessert).
4. followup_resolver — if the user is asking a follow-up, substitution, clarification, or something vague.

Query: {user_input}

Respond with only one word: recipe_finder, instruction_fetcher, pairing_advisor, or followup_resolver.
""")

def classify_intent(user_input):
    prompt = intent_prompt.format(user_input=user_input)
    response = llm.generate_content(prompt)
    return response.text.strip().lower()

# 4. Gemini-based function for all tasks
def gemini_generate(prompt):
    response = llm.generate_content(prompt)
    return response.text.strip()

# 5. LangGraph Nodes
def input_handler(state):
    memory = state["memory"]
    memory.save_context({"input": state["user_input"]}, {"output": "Processing request..."})
    intent = classify_intent(state["user_input"])
    state["intent"] = intent
    return state

def recipe_finder(state):
    user_input = state["user_input"].lower()
    ingredients = [w for w in user_input.replace(",", " ").split() if w not in ["i", "have", "and", "with", "some", "a", "an", "the", "what", "can", "make"]]
    state["last_ingredients"] = ingredients
    prompt = f"""You are a smart cooking assistant. Based on the ingredients: {', '.join(ingredients)}, suggest possible dishes the user can cook. Mention the dish name and a short description."""
    result = gemini_generate(prompt)
    dish_line = result.split("\n")[0].strip().replace("**", "").replace("Dish:", "").strip()
    state["last_recipe"] = dish_line
    state["response"] = f"🍳 {result}"
    return state

def instruction_fetcher(state):
    recipe = state.get("last_recipe")
    if not recipe:
        state["response"] = "Please tell me the ingredients or a dish name first."
        return state
    prompt = f"""How do I cook {recipe}? Provide step-by-step instructions in simple language."""
    result = gemini_generate(prompt)
    state["response"] = f"{result}"
    return state

def pairing_advisor(state):
    recipe = state.get("last_recipe") or "an Indian dish"
    prompt = f"""Suggest a drink or side dish to pair with {recipe}. Explain briefly why it pairs well."""
    result = gemini_generate(prompt)
    state["response"] = f"{result}"
    return state

def followup_resolver(state):
    user_input = state.get("user_input", "").strip()
    last_recipe = state.get("last_recipe")

    if not last_recipe:
        state["response"] = "Please clarify. Ask about recipes, instructions, or pairings."
        return state

    prompt = f"""
You are a helpful cooking assistant. The user previously got the dish: "{last_recipe}". Now they asked: "{user_input}". Based on this, give a helpful and conversational response related to the previous context.
"""
    response = gemini_generate(prompt)
    state["response"] = f"{response}"
    return state

def response_generator(state):
    return state

# 6. Router
def route_intent(state):
    return state.get("intent", "followup_resolver")

# 7. Build LangGraph
builder = StateGraph(RecipeAgentState)
builder.add_node("input_handler", input_handler)
builder.add_node("recipe_finder", recipe_finder)
builder.add_node("instruction_fetcher", instruction_fetcher)
builder.add_node("pairing_advisor", pairing_advisor)
builder.add_node("followup_resolver", followup_resolver)
builder.add_node("response_generator", response_generator)

builder.set_entry_point("input_handler")

builder.add_conditional_edges("input_handler", route_intent, {
    "recipe_finder": "recipe_finder",
    "instruction_fetcher": "instruction_fetcher",
    "pairing_advisor": "pairing_advisor",
    "followup_resolver": "followup_resolver"
})

builder.add_edge("recipe_finder", "response_generator")
builder.add_edge("instruction_fetcher", "response_generator")
builder.add_edge("pairing_advisor", "response_generator")
builder.add_edge("followup_resolver", "response_generator")
builder.add_edge("response_generator", END)

recipe_graph = builder.compile()

# 8. Execution
if __name__ == "__main__":
    print("Recipe Agent Ready!")
    state = initialize_state("")

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        if user_query.strip():
            state["user_input"] = user_query
            final_state = recipe_graph.invoke(state)

            # Keep context
            state["last_recipe"] = final_state.get("last_recipe", state.get("last_recipe"))
            state["last_ingredients"] = final_state.get("last_ingredients", state.get("last_ingredients"))
            state["memory"] = final_state.get("memory", state.get("memory"))

            print("\n", final_state["response"])
