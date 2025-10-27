import sys
import os

# Adiciona o backend ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.main import app

__all__ = ["app"]