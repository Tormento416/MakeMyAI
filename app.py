"""
Main Entry Point Launcher for SLM/LLM Architect & Builder Studio Desktop GUI App.
"""

import os
import sys

# Ensure package root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slm_builder.gui.main_window import main

if __name__ == "__main__":
    main()
