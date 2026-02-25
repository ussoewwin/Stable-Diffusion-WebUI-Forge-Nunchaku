import transformers
import inspect
import sys

def dump_module_content():
    try:
        from transformers.models.florence2 import configuration_florence2
        print(f"Content of {configuration_florence2.__name__}:")
        for name in dir(configuration_florence2):
            print(name)
            
        # Also print Florence2Config source to see how it uses text config
        if hasattr(configuration_florence2, "Florence2Config"):
             print("\n--- Florence2Config Source ---")
             print(inspect.getsource(configuration_florence2.Florence2Config))
             
    except ImportError as e:
        print(f"Failed to import configuration_florence2: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    dump_module_content()
    dump_source(Florence2ForConditionalGeneration.forward, "Florence2ForConditionalGeneration.forward")
