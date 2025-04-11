#!/usr/bin/env python3
"""
Documentation generation script for the Cuttle Bot project.
This script generates HTML documentation using pdoc.
"""

import os
import sys
import pdoc
from pathlib import Path

def generate_docs():
    """
    Generate documentation for the project.
    """
    # Define the output directory
    output_dir = Path("docs")
    output_dir.mkdir(exist_ok=True)
    
    # Define the modules to document
    modules = [
        "game",
        "game.game",
        "game.game_state",
        "game.card",
        "game.action",
        "game.ai_player",
        "game.input_handler",
        "game.serializer",
        "game.utils",
        "main",
    ]
    
    # Generate documentation
    pdoc.render.configure(
        docformat="google",  # Use Google-style docstrings
        show_source=True,    # Show source code
    )
    
    # Generate documentation for each module
    for module in modules:
        try:
            pdoc.pdoc(
                *modules,
                output_directory=output_dir,
            )
            print(f"Generated documentation for {module}")
        except Exception as e:
            print(f"Error generating documentation for {module}: {e}")
    
    print(f"\nDocumentation generated in {output_dir.absolute()}")
    print("You can view the documentation by opening docs/index.html in your browser")

if __name__ == "__main__":
    generate_docs() 