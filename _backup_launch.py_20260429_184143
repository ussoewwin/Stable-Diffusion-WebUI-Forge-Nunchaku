import os
import sys

# NumPy 2.2.6 compatibility
os.environ.setdefault("NUMPY_WARN_IF_NO_MEM_POLICY", "0")

# NumPy 2.x compatibility patch for opencv-python
# In NumPy 2.x, numpy.core is deprecated and moved to numpy._core
# This patch ensures numpy.core.multiarray is available before cv2 imports it
def patch_numpy_core():
    try:
        import numpy as np
        
        # In NumPy 2.x, numpy.core is deprecated and moved to numpy._core
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
            except (ImportError, AttributeError):
                # Fallback: try to ensure numpy.core.multiarray exists
                try:
                    import numpy.core.multiarray
                    if not hasattr(np.core.multiarray, '_ARRAY_API'):
                        class _ArrayAPI:
                            pass
                        np.core.multiarray._ARRAY_API = _ArrayAPI()
                    sys.modules['numpy.core.multiarray'] = np.core.multiarray
                except Exception:
                    pass
        else:
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
    except Exception:
        # If numpy is not available, just continue
        pass

# Apply the patch before any cv2 imports
patch_numpy_core()

from modules import launch_utils

args = launch_utils.args
python = launch_utils.python
git = launch_utils.git
index_url = launch_utils.index_url
dir_repos = launch_utils.dir_repos

if args.uv or args.uv_symlink:
    from modules_forge.uv_hook import patch
    patch(args.uv_symlink)

git_tag = launch_utils.git_tag

run = launch_utils.run
is_installed = launch_utils.is_installed
repo_dir = launch_utils.repo_dir

run_pip = launch_utils.run_pip
check_run_python = launch_utils.check_run_python
git_clone = launch_utils.git_clone
git_pull_recursive = launch_utils.git_pull_recursive
list_extensions = launch_utils.list_extensions
run_extension_installer = launch_utils.run_extension_installer
prepare_environment = launch_utils.prepare_environment
start = launch_utils.start


def main():
    if args.dump_sysinfo:
        filename = launch_utils.dump_sysinfo()

        print(f"Sysinfo saved as {filename}. Exiting...")

        exit(0)

    launch_utils.startup_timer.record("initial startup")

    with launch_utils.startup_timer.subcategory("prepare environment"):
        if not args.skip_prepare_environment:
            prepare_environment()

    if args.forge_ref_a1111_home:
        launch_utils.configure_a1111_reference(args.forge_ref_a1111_home)
    if args.forge_ref_comfy_home:
        launch_utils.configure_comfy_reference(args.forge_ref_comfy_home)

    start()


if __name__ == "__main__":
    main()
