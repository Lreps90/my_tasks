import os
from tkinter import Tk, filedialog, simpledialog
from docx import Document
from tkinter import messagebox

def select_folder(prompt):
    root = Tk()
    root.withdraw() # Hide the Tkinter root window
    folder_selected = filedialog.askdirectory(title=prompt)
    root.destroy()
    return folder_selected

def txt_to_docx(input_folder, output_folder):
    # Check if the folder exists
    if not os.path.exists(input_folder):
        messagebox.showerror("Error", f"The folder '{input_folder}' does not exist.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory '{output_folder}'.")

    # Iterate through all files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            # Create a Word document
            doc = Document()
            txt_file_path = os.path.join(input_folder, filename)

            # Try opening the text file with different encodings
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                try:
                    with open(txt_file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                        doc.add_paragraph(content)
                        break  # If successful, break out of the loop
                except UnicodeDecodeError:
                    continue  # Try the next encoding
            else:
                # If all encodings fail, print an error message and skip this file
                print(f"Failed to decode '{filename}'. File skipped.")
                continue

            # Save the Word document with the same name as the text file in the output directory
            docx_filename = filename[:-4] + ".docx"
            docx_file_path = os.path.join(output_folder, docx_filename)
            doc.save(docx_file_path)
            print(f"Converted '{filename}' to '{docx_filename}' in the output directory.")


def main():
    input_folder = select_folder("Select Input Directory")
    if not input_folder:
        messagebox.showinfo("Info", "Input directory selection was cancelled.")
        return

    output_folder = select_folder("Select Output Directory")
    if not output_folder:
        messagebox.showinfo("Info", "Output directory selection was cancelled.")
        return

    txt_to_docx(input_folder, output_folder)
    messagebox.showinfo("Success", "All text files have been converted to Word documents.")

if __name__ == "__main__":
    main()
