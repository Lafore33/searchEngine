import pdfplumber as plumber

class DocParser:

    def parse_to_string(self, filename: str) -> str:
        with plumber.open(filename) as pdf:
            return '\n\n'.join(page.extract_text() for page in pdf.pages)