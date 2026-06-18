# =============================================================================
# INVESTMENT BRAIN AGENT — Phase 1 (Google Colab)
# =============================================================================


# =============================================================================
# Install All Dependencies
# =============================================================================
"""
Install all required packages for the Investment Brain Agent.
Run this cell once at the start of each Colab session.
"""

import subprocess
import sys
import re # Added for extracting package names from version specifiers

def install_dependencies():
    """Install all required Python packages for the project."""
    packages = [
        "google-generativeai>=0.7.0",
        "sentence-transformers>=2.7.0",
        "chromadb==0.4.24",
        "pymupdf>=1.24.0",
        "pdfplumber>=0.11.0",
        "rank_bm25>=0.2.2",
        "numpy<2.0",
        "tqdm>=4.66.0",
        "tenacity>=8.3.0",
        "reportlab",
        "mistralai>=2.4.9",
        "langchain-core>=0.3.0",
        "langchain-mistralai>=0.2.0",
        "langchain-google-genai>=2.0.0"
    ]
    for pkg_spec in packages:
        # Extract the base package name (e.g., 'google-generativeai' from 'google-generativeai>=0.7.0')
        pkg_name = re.split(r'[<=>~]', pkg_spec)[0].strip()
        try:
            # Install the package silently
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_spec, "-q"])

            # If successful, get the installed version
            try:
                show_output = subprocess.check_output(
                    [sys.executable, "-m", "pip", "show", pkg_name],
                    stderr=subprocess.STDOUT
                ).decode()
                version_match = re.search(r'^Version: (.+)$', show_output, re.MULTILINE)
                if version_match:
                    version = version_match.group(1)
                    print(f"Package '{pkg_name}' installed successfully (version {version}).")
                else:
                    print(f"Package '{pkg_name}' installed successfully, but installed version could not be determined.")
            except subprocess.CalledProcessError:
                print(f"Package '{pkg_name}' installed successfully, but 'pip show' failed to retrieve version.")

        except subprocess.CalledProcessError as e:
            # If installation fails, print the error output from pip
            error_message = e.output.decode().strip() if e.output else "Unknown error during installation."
            print(f"Error installing package '{pkg_spec}': {error_message}")
        except Exception as e:
            # Catch any other unexpected errors during the process
            print(f"An unexpected error occurred for package '{pkg_spec}': {e}")

install_dependencies()

# ----------------------------------------------------------------------------
# File Name: packages.py
# Purpose: Install every third-party library the project needs.
# Key Classes: None
# Key Functions: install_dependencies() → None
# Key Constants/Config: packages list (hardcoded here only — bootstrapping concern)
# Imports exported: None
# Depends on: None
# Critical notes: Run once per session. Order does not matter.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------