import os
import sys

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.run_pipeline import run_pipeline_for_text

def run_integrate_full_stack_mode(text: str):
    return run_pipeline_for_text(text, mode='full_stack')