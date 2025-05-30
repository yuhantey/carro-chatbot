from PyPDF2 import PdfReader

def extract_faq_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        full_text += text + "\n"
    return full_text
