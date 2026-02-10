import PyPDF2


def extract_page(pdf_path, page_number, output_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        writer = PyPDF2.PdfWriter()

        if page_number < 0 or page_number >= len(reader.pages):
            print("Invalid page number.")
            return

        writer.add_page(reader.pages[page_number])

        with open(output_path, 'wb') as output_file:
            writer.write(output_file)
            print(f"Page {page_number + 1} extracted and saved as {output_path}")


# Example usage:
pdf_path = r'C:\Users\Lreps\OneDrive - Heathside School\Computer Science\A-Level\OCR_A-LEVEL_SPECIFICATION.pdf'  # Replace 'input.pdf' with the path to your PDF file
output_path = r'C:\Users\Lreps\OneDrive - Heathside School\Computer Science\A-Level\analysis_MS.pdf'  # Replace 'output.pdf' with the desired output path
page_number = 26  # Page numbering starts from 0, so page 27 is at index 26

extract_page(pdf_path, page_number, output_path)
