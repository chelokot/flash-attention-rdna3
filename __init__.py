from pathlib import Path
import sys


repository_root = Path(__file__).resolve().parent

if __package__:
    from .fa_rdna3.comfyui import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    loaded_package = sys.modules.get("fa_rdna3")
    if loaded_package is not None:
        loaded_path = Path(loaded_package.__file__).resolve()
        if repository_root not in loaded_path.parents:
            raise RuntimeError(
                "A different fa_rdna3 package is already loaded; restart ComfyUI "
                "before loading this custom node"
            )

    sys.path.insert(0, str(repository_root))
    try:
        from fa_rdna3.comfyui import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    finally:
        sys.path.pop(0)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
