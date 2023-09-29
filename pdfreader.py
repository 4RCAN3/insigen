import PyPDF2

def readPdf(filename: str) -> list:

    with open(filename, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        pages = [reader.pages[i].extract_text() for i in range(len(reader.pages))]

        f.close()

    return pages

print(readPdf('whitepaper.pdf'))