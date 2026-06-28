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
from logger import get_logger


def run():
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

    logger = get_logger("FILELOADER", filename)

    try:
        logger.event(
            f"{filename} : File stored in temporary storage succesfully",
        )
        result = upload_file(file_bytes, filename, logger)
    except FilenameValidationError as exc:
        print(f"Filename rejected: {exc}")
    except DuplicateFileError as exc:
        print(f"Upload rejected: {exc}")
    if result is not None and not isinstance(result, str):
        try: 
            insert_upload_record(result)
        except DatabaseInsertError as exc:
            logger.event(f"File uploaded but records were not appended to Database : {exc}")
    else:
        if isinstance(result, str):
            return result
        else:
            return None
    


if __name__ == "__main__":
    asyncio.run(run())