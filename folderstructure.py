# =============================================================================
# CELL 2 — Mount Drive + Create Folder Structure
# =============================================================================
"""
Mount Google Drive and create the canonical folder structure under /brain/.
Creates all required directories if they do not already exist.
"""

import os

BRAIN_BASE = os.path.dirname(os.path.abspath(__file__))
#BRAIN_BASE = "/workspaces/brain/"
def get_base_path() -> str:
  """Returns Base Path Address"""
  return BRAIN_BASE

def mount_drive():
    """Mount Google Drive in Colab environment."""
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=False)
        print("Google Drive mounted.")
    except ImportError:
        print("[Cell 2] Not running in Colab — skipping Drive mount.")

def create_folder_structure(base_path: str = "/content/drive/MyDrive/brain") -> None:
    """
    Create the full folder hierarchy under base_path.

    Parameters
    ----------
    base_path : str
        Root path for all project files.
    """
    folders = [
        base_path,
        os.path.join(base_path, "uploads"),
        os.path.join(base_path, "chroma_db"),
        os.path.join(base_path, "database"),
        os.path.join(base_path, "logs"),
    ]
    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)
            print(f"✅ Created/Verified: {folder}")
        except PermissionError as e:
            print(f"❌ Permission denied - cannot create folder: {folder}\n   Error: {e}")
        except OSError as e:
            print(f"❌ OS Error - failed to create folder: {folder}\n   Error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error creating folder: {folder}\n   Error: {e}")
    print(f"📁 Folder structure verified under: {base_path}")

#mount_drive()

#create_folder_structure(BRAIN_BASE)

# ----------------------------------------------------------------------------
# File Name: folderstructure.py
# Purpose: Mount Google Drive and ensure all project directories exist.
# Key Classes: None
# Key Functions: mount_drive() → None, create_folder_structure(base_path) → None
# Key Constants/Config: BRAIN_BASE
# Imports exported: BRAIN_BASE (used by Cell 3 config)
# Depends on: None
# Critical notes: BRAIN_BASE must match Config.BASE_PATH in Cell 3.
#   In non-Colab environments the Drive mount is skipped gracefully.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------