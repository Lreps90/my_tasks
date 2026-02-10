import os
import shutil
from PyPDF2 import PdfReader, PdfWriter

def split_pdf(input_pdf_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the input PDF file
    pdf_reader = PdfReader(input_pdf_path)
    total_pages = len(pdf_reader.pages)

    for i in range(0, total_pages, 10):
        # Create a new PDF writer for each batch of 6 pages
        pdf_writer = PdfWriter()

        # Add 6 pages from the input PDF to the new PDF
        for page_num in range(i, min(i + 6, total_pages)):
            pdf_writer.add_page(pdf_reader.pages[page_num])

        # Save the new PDF to a separate file
        output_pdf_path = os.path.join(output_folder, f'split_{i // 6 + 1}.pdf')
        with open(output_pdf_path, 'wb') as output_file:
            pdf_writer.write(output_file)

if __name__ == "__main__":
    input_pdf_path = 'input.pdf'  # Replace with the path to your input PDF
    output_folder = 'output'  # Replace with the name of the output folder



    split_pdf(r'C:\Users\Lreps\OneDrive - Heathside School\Maths\A-Level\Madasmaths\binomial_distribution_part2.pdf', r'C:\Users\Lreps\OneDrive - Heathside School\Maths\A-Level\Madasmaths')
    print(f'PDF split into separate files in the "{output_folder}" folder.')

