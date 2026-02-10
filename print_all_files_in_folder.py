import os
import win32com.client

def print_word_files_in_folder(folder_path, printer_name):
    """
    Prints all Microsoft Word files in the specified folder with the settings:
    - 2 pages to 1 page
    - Colour printer

    Args:
        folder_path (str): Path to the folder containing Word files to print.
        printer_name (str): Name of the colour printer to use.
    """
    if not os.path.exists(folder_path):
        print("The specified folder does not exist.")
        return

    files = [f for f in os.listdir(folder_path) if f.endswith(".doc") or f.endswith(".docx")]

    if not files:
        print("No Word files found in the folder.")
        return

    word_app = win32com.client.Dispatch("Word.Application")
    word_app.Visible = False

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        try:
            doc = word_app.Documents.Open(file_path)
            # Set the printer
            word_app.ActivePrinter = printer_name
            # Print with specific settings
            doc.PrintOut(
                Range=0,  # Print all pages
                Item=0,   # Print document content
                Copies=1,
                Pages=None,
                PageType=0,
                PrintToFile=False,
                Collate=True,
                Background=True,
                Append=False,
                ManualDuplexPrint=False
            )
            print(f"Sent '{file_name}' to the printer.")
            doc.Close(False)
        except Exception as e:
            print(f"Failed to print '{file_name}': {e}")

    word_app.Quit()

if __name__ == "__main__":
    folder = r"C:\Users\Lreps\OneDrive - Heathside School\Computer Science\GCSE\Assessments\2024-2025\QLA"
    printer = "Papercut_Color_Print"
    print_word_files_in_folder(folder, printer)
