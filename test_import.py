import sys
print("Python Executable:", sys.executable)
try:
    import langchain
    print(f"langchain file: {langchain.__file__}")
    import langchain.chains
    print("langchain.chains imported")
    from langchain.chains import RetrievalQA
    print("RetrievalQA imported")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
