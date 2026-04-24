
"""
Color Blind Accessibility server module
"""

"""from .environment import CBAEnvironment

__all__ = ["CBAEnvironment"]"""
import os
import sys
# Force python to load the local 'src' directory BEFORE importing environment!
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from .environment import CBAEnvironment

__all__ = ["CBAEnvironment"]