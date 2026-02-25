import inspect
from transformers import Florence2Processor

try:
    source = inspect.getsource(Florence2Processor.__call__)
    print(source)
except Exception as e:
    print(f"Could not get source: {e}")
