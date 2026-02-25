import torch
from transformers import Florence2Processor, Florence2ForConditionalGeneration, Florence2Config
from PIL import Image
import numpy as np

def run_test():
    print("Testing Florence2 API Compatibility with transformers 5.2...")

    # Mock inputs
    dummy_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    dummy_text = "<OD>"
    
    # 1. Test Processor Loading (using dummy config/processor if needed, or just checking class)
    try:
        # We don't have model weights, so we can't fully load unless we download.
        # But we can inspect the class methods which is what we need.
        proc_cls = Florence2Processor
        model_cls = Florence2ForConditionalGeneration
        
        print(f"Processor class: {proc_cls}")
        print(f"Model class: {model_cls}")

        # Check processor __call__ signature for do_rescale
        import inspect
        sig = inspect.signature(proc_cls.__call__)
        print(f"Processor.__call__ params: {list(sig.parameters.keys())}")
        
        if 'do_rescale' in sig.parameters or 'kwargs' in sig.parameters:
             print("Processor.__call__ likely supports do_rescale (via arg or kwargs)")
        else:
             print("WARNING: Processor.__call__ might NOT support do_rescale")

        # Check post_process_generation signature
        if hasattr(proc_cls, 'post_process_generation'):
             sig_pp = inspect.signature(proc_cls.post_process_generation)
             print(f"post_process_generation params: {list(sig_pp.parameters.keys())}")
        else:
             print("ERROR: Processor missing post_process_generation")

        # Check model generate signature for pixel_values
        sig_gen = inspect.signature(model_cls.generate)
        print(f"Model.generate params: {list(sig_gen.parameters.keys())}")
        
        if 'pixel_values' in sig_gen.parameters or 'kwargs' in sig_gen.parameters:
             print("Model.generate likely supports pixel_values")
        else:
             print("WARNING: Model.generate might NOT support pixel_values directly")

    except Exception as e:
        print(f"Test failed with error: {e}")

if __name__ == "__main__":
    run_test()
