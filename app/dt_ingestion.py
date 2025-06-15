import os
import re
from PyPDF2 import PdfReader
import markdown
class Doc_processor:
    def __init__(self):
        pass 
        
    def _loadpdf(self,filepath):
        text=""
        try:
            with open(filepath,'rb') as file:
                read=PdfReader(file)
                for p in read.pages:
                    text+=p.extract_text() + "\n"
        except Exception as e:
            print(f'Error Loading :{e}')
        
        return text

    def _loadtxt(self,filepath):
        text=""
        try:
            with open(filepath, "r") as file:
                return file.read()    
        except Exception as e:
            print(f'Error Loading :{e}')
    
    def _loadmarkdown(self ,filepath):
        text=""
        try:
            with open(filepath,'r',encoding='utf-8') as file:
                text=file.read()
                html=markdown.markdown(text)
                text=re.sub('<[^<]+?', '',html)
                return text
        except Exception as e:
            print(f'Error Loading :{e}')
    
    def load_document(self,filepath):
        extension=os.path.splitext(filepath)[1].lower()
        if extension =='.md':
            return self._loadmarkdown(filepath)
        elif extension =='.pdf':
            return self._loadpdf(filepath)
        elif extension=='.txt':
            return self._loadtxt(filepath)
        else:
            raise ValueError('File Format Not Supported. (Supported are only pdf,txt,md)')
        