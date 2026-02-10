# import tkinter as tk
# from tkinter import filedialog, messagebox
# import pandas as pd
# import os
#
# def select_file():
#     global input_file_path
#     input_file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls")])
#     if input_file_path:
#         file_label.config(text=os.path.basename(input_file_path))
#
# def select_output_folder():
#     global output_folder_path
#     output_folder_path = filedialog.askdirectory()
#     if output_folder_path:
#         folder_label.config(text=output_folder_path)
#
# def split_excel_to_text():
#     if not input_file_path or not output_folder_path:
#         messagebox.showerror("Error", "Please select both an Excel file and an output folder.")
#         return
#
#     df = pd.read_excel(input_file_path)
#
#     if not os.path.exists(output_folder_path):
#         os.makedirs(output_folder_path)
#
#     for index, row in df.iterrows():
#         name = str(row['Name']).strip().replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
#         filename = f"{name}.txt"
#         with open(os.path.join(output_folder_path, filename), 'w', encoding='utf-8') as file:
#             for column in df.columns:
#                 # Strip whitespace from the start and end of the string representation of the value
#                 value = str(row[column]).strip()
#                 # Write the column and its value, followed by an extra newline character for spacing
#                 file.write(f"{column}: {value}\n\n")  # Added an extra \n here for the blank line
#
#     messagebox.showinfo("Success", "Excel file has been successfully split into text files.")
#
# # Set up the GUI
# root = tk.Tk()
# root.title("Excel to Text Splitter")
#
# input_file_path = ''
# output_folder_path = ''
#
# tk.Button(root, text="Select Excel File", command=select_file).pack()
# file_label = tk.Label(root, text="No file selected")
# file_label.pack()
#
# tk.Button(root, text="Select Output Folder", command=select_output_folder).pack()
# folder_label = tk.Label(root, text="No folder selected")
# folder_label.pack()
#
# tk.Button(root, text="Process", command=split_excel_to_text).pack()
#
# root.mainloop()


