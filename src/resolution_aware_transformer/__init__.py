"""resolution_aware_transformer: PyTorch implementation of a Resolution Aware Transformer, designed to take in multiple scales of fixed-resolution images (such as microscopy or medical imaging), with the goal of producing embeddings informed by global context and local details. Utilizes rotary spatial embeddings (RoSE) and spatial grouping attention.

A Python package for pytorch implementation of a resolution aware transformer, designed to take in multiple scales of fixed-resolution images (such as microscopy or medical imaging), with the goal of producing embeddings informed by global context and local details. utilizes rotary spatial embeddings (rose) and spatial grouping attention..
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("resolution-aware-transformer")
except PackageNotFoundError:
    __version__ = "0.1.0"

__author__ = "Jeff Rhoades"
__email__ = "rhoadesj@alumni.harvard.edu"

from .resolution_aware_transformer import ResolutionAwareTransformer

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "ResolutionAwareTransformer",
]
