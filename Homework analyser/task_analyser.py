import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd


def browse_file():
    file_path = filedialog.askopenfilename(
        title="Select Excel file",
        filetypes=[("Excel files", "*.xlsx;*.xls"), ("All files", "*.*")]
    )
    if file_path:
        process_file(file_path)


def process_file(file_path):
    try:
        # Read the Excel file using the second row as headers (header=1).
        df = pd.read_excel(file_path, header=1)
    except Exception as e:
        messagebox.showerror("Error", f"Error reading file '{file_path}': {e}")
        return

    # Ensure required columns exist.
    if 'Status' not in df.columns or 'Full Name' not in df.columns:
        messagebox.showerror("Error", "The selected file must contain 'Status' and 'Full Name' columns.")
        return

    # Filter rows where the 'Status' is NOT 'turned in' (case-insensitive).
    incomplete_df = df[~df['Status'].str.strip().str.lower().eq("turned in")]

    # Extract unique student names from the "Full Name" column.
    incomplete_students = incomplete_df["Full Name"].dropna().unique()

    # Display results.
    if len(incomplete_students) == 0:
        messagebox.showinfo("Homework Completion", "All students have completed the homework.")
    else:
        student_list = "\n".join(incomplete_students)
        messagebox.showinfo("Incomplete Homework", f"Students who have not completed the homework:\n\n{student_list}")


def main():
    # Initialise the Tkinter root and hide the main window.
    root = tk.Tk()
    root.withdraw()

    # Open the file browser dialog.
    browse_file()

    # Close the Tkinter application after processing.
    root.destroy()


if __name__ == '__main__':
    main()
