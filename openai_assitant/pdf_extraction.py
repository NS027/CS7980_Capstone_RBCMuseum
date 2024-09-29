import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# Extract text from your PDF file
pdf_text = extract_text_from_pdf("data/Haida_bracelet.pdf")
print(f"Extracted PDF text with length: {len(pdf_text)} characters")
