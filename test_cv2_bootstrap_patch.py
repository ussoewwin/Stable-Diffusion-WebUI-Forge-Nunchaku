#!/usr/bin/env python
"""Test script for patching cv2 bootstrap to work with numpy 2.x"""
import os
import sys
import importlib

# Set environment variable
os.environ.setdefault("NUMPY_WARN_IF_NO_MEM_POLICY", "0")

import numpy as np
print(f"numpy version: {np.__version__}")

# Patch numpy.core before cv2 import
# cv2's __init__.py tries to import numpy.core.multiarray directly at line 12
try:
    # In numpy 2.x, numpy.core is deprecated and moved to numpy._core
    # We need to ensure numpy.core is available before cv2 imports it
    if not hasattr(np, 'core') or 'numpy.core.multiarray' not in sys.modules:
        try:
            # Try to import from numpy._core and map it to numpy.core
            import numpy._core.multiarray as _multiarray
            import numpy._core._multiarray_umath as _multiarray_umath
            
            # Create numpy.core namespace if it doesn't exist
            if not hasattr(np, 'core'):
                class _Core:
                    pass
                np.core = _Core()
            
            # Map numpy._core to numpy.core
            np.core.multiarray = _multiarray
            np.core._multiarray_umath = _multiarray_umath
            
            # Ensure _ARRAY_API exists
            if hasattr(_multiarray_umath, '_ARRAY_API'):
                np.core.multiarray._ARRAY_API = _multiarray_umath._ARRAY_API
            elif not hasattr(np.core.multiarray, '_ARRAY_API'):
                class _ArrayAPI:
                    pass
                np.core.multiarray._ARRAY_API = _ArrayAPI()
            
            # Register in sys.modules so cv2 can find it
            # This is critical - cv2's __init__.py does: import numpy.core.multiarray
            sys.modules['numpy.core'] = np.core
            sys.modules['numpy.core.multiarray'] = np.core.multiarray
            sys.modules['numpy.core._multiarray_umath'] = np.core._multiarray_umath
            
            print("Successfully patched numpy.core from numpy._core")
            print(f"sys.modules has numpy.core.multiarray: {'numpy.core.multiarray' in sys.modules}")
        except (ImportError, AttributeError) as e:
            print(f"Failed to patch from numpy._core: {e}")
            # Fallback: try to ensure numpy.core.multiarray exists
            try:
                import numpy.core.multiarray
                if not hasattr(np.core.multiarray, '_ARRAY_API'):
                    class _ArrayAPI:
                        pass
                    np.core.multiarray._ARRAY_API = _ArrayAPI()
                sys.modules['numpy.core.multiarray'] = np.core.multiarray
                print("Using numpy.core.multiarray directly")
            except Exception as e2:
                print(f"Failed to import numpy.core.multiarray: {e2}")
    else:
        print("numpy.core.multiarray already in sys.modules")
        # Ensure _ARRAY_API exists
        if hasattr(np.core, 'multiarray') and not hasattr(np.core.multiarray, '_ARRAY_API'):
            try:
                from numpy.core import _multiarray_umath
                if hasattr(_multiarray_umath, '_ARRAY_API'):
                    np.core.multiarray._ARRAY_API = getattr(_multiarray_umath, '_ARRAY_API')
                else:
                    class _ArrayAPI:
                        pass
                    np.core.multiarray._ARRAY_API = _ArrayAPI()
            except (ImportError, AttributeError):
                class _ArrayAPI:
                    pass
                np.core.multiarray._ARRAY_API = _ArrayAPI()
except Exception as e:
    print(f"Error in patching: {e}")
    import traceback
    traceback.print_exc()

# Test if numpy.core.multiarray can be imported (like cv2 does)
print("\nTesting direct import of numpy.core.multiarray (like cv2 does)...")
try:
    import numpy.core.multiarray
    print("Successfully imported numpy.core.multiarray")
    print(f"Has _ARRAY_API: {hasattr(numpy.core.multiarray, '_ARRAY_API')}")
except Exception as e:
    print(f"Failed to import numpy.core.multiarray: {e}")
    import traceback
    traceback.print_exc()

# Try to import cv2
print("\nAttempting to import cv2...")
try:
    import cv2
    print(f"cv2 imported successfully! version: {cv2.__version__}")
except (ImportError, AttributeError) as e:
    print(f"Failed to import cv2: {e}")
    import traceback
    traceback.print_exc()
