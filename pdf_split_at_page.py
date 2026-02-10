import PyPDF2
import tkinter as tk
from tkinter import filedialog, messagebox
import os


def split_pdf(input_pdf, split_at_page, output_pdf1, output_pdf2):
    try:
        with open(input_pdf, 'rb') as infile:
            reader = PyPDF2.PdfReader(infile)

            if split_at_page < 1 or split_at_page > len(reader.pages):
                raise ValueError(f"Invalid split page: {split_at_page}. It must be between 1 and {len(reader.pages)}.")

            writer1 = PyPDF2.PdfWriter()
            writer2 = PyPDF2.PdfWriter()

            for i in range(split_at_page):
                writer1.add_page(reader.pages[i])

            for i in range(split_at_page, len(reader.pages)):
                writer2.add_page(reader.pages[i])

            with open(output_pdf1, 'wb') as outfile1:
                writer1.write(outfile1)

            with open(output_pdf2, 'wb') as outfile2:
                writer2.write(outfile2)

            messagebox.showinfo("Success", f"PDF split successfully.\nSaved as '{output_pdf1}' and '{output_pdf2}'.")
    except Exception as e:
        messagebox.showerror("Error", str(e))


def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if file_path:
        entry_file_path.delete(0, tk.END)
        entry_file_path.insert(0, file_path)


def split_button_clicked():
    input_pdf = entry_file_path.get()
    split_at_page = entry_page_number.get()

    if not input_pdf or not split_at_page:
        messagebox.showwarning("Input Error", "Please select a PDF file and enter a page number.")
        return

    try:
        split_at_page = int(split_at_page)
    except ValueError:
        messagebox.showwarning("Input Error", "Please enter a valid page number.")
        return

    base_name = os.path.splitext(os.path.basename(input_pdf))[0]
    directory = os.path.dirname(input_pdf)

    output_pdf1 = os.path.join(directory, f"{base_name}_part1.pdf")
    output_pdf2 = os.path.join(directory, f"{base_name}_part2.pdf")

    split_pdf(input_pdf, split_at_page, output_pdf1, output_pdf2)


# GUI setup
root = tk.Tk()
root.title("PDF Splitter")

frame = tk.Frame(root)
frame.pack(pady=20, padx=20)

label_file_path = tk.Label(frame, text="Select PDF file:")
label_file_path.grid(row=0, column=0, padx=5, pady=5)

entry_file_path = tk.Entry(frame, width=50)
entry_file_path.grid(row=0, column=1, padx=5, pady=5)

button_browse = tk.Button(frame, text="Browse", command=browse_file)
button_browse.grid(row=0, column=2, padx=5, pady=5)

label_page_number = tk.Label(frame, text="Split at page number:")
label_page_number.grid(row=1, column=0, padx=5, pady=5)

entry_page_number = tk.Entry(frame, width=10)
entry_page_number.grid(row=1, column=1, padx=5, pady=5)

button_split = tk.Button(frame, text="Split PDF", command=split_button_clicked)
button_split.grid(row=2, columnspan=3, pady=10)

root.mainloop()
