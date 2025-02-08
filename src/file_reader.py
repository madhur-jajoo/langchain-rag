# from langchain_community.document_loaders import PyPDFLoader
import PyPDF2

def process_file(file):
    content = ""
    if file is not None:
        pdf_loader = PyPDF2.PdfReader(file)
        for page in range(len(pdf_loader.pages)):
            content += pdf_loader.pages[page].extract_text()

    return content