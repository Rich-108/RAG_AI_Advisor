from langchain_ollama import OllamaLLM

# 1. Initialize the model (using the 1B version for your i3)
print("--- Initializing Llama 3.2 1B ---")
llm = OllamaLLM(model="llama3.2:1b")

# 2. Ask a simple test question
question = "Hello! I am a Computer Science Professor. Are you ready to help me?"

print(f"Question: {question}")
print("\n--- AI is thinking... ---\n")

# 3. Get the response
try:
    response = llm.invoke(question)
    print("AI Response:")
    print(response)
    print("\n--------------------------")
    print("SUCCESS: Your Day 1 setup is 100% complete!")
except Exception as e:
    print(f"ERROR: Something went wrong. Make sure Ollama is running. \n{e}")