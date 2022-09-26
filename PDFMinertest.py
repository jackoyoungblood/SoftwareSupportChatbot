from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import PDFPageAggregator
from pdfminer3.converter import TextConverter
import io

resource_manager = PDFResourceManager()
fake_file_handle = io.StringIO()
converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
page_interpreter = PDFPageInterpreter(resource_manager, converter)

txtfile = open('c:\dev\IRM Web Client User Guide (Legal Version) 10.3.3.pdf.txt', 'w')

with open('c:\dev\IRM Web Client User Guide (Legal Version) 10.3.3.pdf', 'rb') as fh:

    for page in PDFPage.get_pages(fh,
                                  caching=True,
                                  check_extractable=True):
        
        page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()
        n = txtfile.write(text)

# close open handles
converter.close()
fake_file_handle.close()


txtfile.close()
print(text)