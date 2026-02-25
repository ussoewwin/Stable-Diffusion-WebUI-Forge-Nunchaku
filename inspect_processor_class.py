import inspect
from transformers.models.florence2 import processing_florence2

print("Source for processing_florence2:")
try:
    print(inspect.getsource(processing_florence2.Florence2Processor))
except Exception as e:
    print(f"Error: {e}")
