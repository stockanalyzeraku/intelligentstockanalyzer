import asyncio
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

from codebase.fileloader.fileloader import upload_file 
from codebase.fileloader.exceptions import (
    FilenameValidationError,
    DuplicateFileError,
    DatabaseInsertError
)
from codebase.fileloader.db import insert_upload_record


async def run():
    # Hide the empty root tkinter window, only show the file dialog
    root = tk.Tk()
    root.withdraw()

    file_path_str = filedialog.askopenfilename(
        title="Select a PDF file to upload",
        filetypes=[("PDF files", "*.pdf")],
    )
    root.destroy()

    if not file_path_str:
        print("No file selected.")
        return

    file_path = Path(file_path_str)
    filename = file_path.name
    file_bytes = file_path.read_bytes()

    try:
        result = await upload_file(file_bytes, filename)
        print(result)
    except FilenameValidationError as exc:
        print(f"Filename rejected: {exc}")
    except DuplicateFileError as exc:
        print(f"Upload rejected: {exc}")
    try: 
        insert_upload_record(result)
    except DatabaseInsertError as exc:
        print(f"Inser to columns Failed :{exc}")


if __name__ == "__main__":
    asyncio.run(run())