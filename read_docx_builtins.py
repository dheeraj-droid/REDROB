import zipfile
import xml.etree.ElementTree as ET
import sys
import os

def read_docx(path):
    try:
        with zipfile.ZipFile(path) as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.XML(xml_content)
            
            # Namespace for Word XML
            WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
            PARA = WORD_NAMESPACE + 'p'
            TEXT = WORD_NAMESPACE + 't'
            
            texts = []
            for node in tree.iter(PARA):
                para_text = ''.join(node.itertext())
                if para_text:
                    texts.append(para_text)
            
            return '\n'.join(texts)
    except Exception as e:
        return f"Error reading docx: {e}"

if __name__ == "__main__":
    path = r"c:\Users\dheeraj reddy\OneDrive\Desktop\redrob\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\redrob_signals_doc.docx"
    if os.path.exists(path):
        print("Reading DOCX content:")
        print(read_docx(path))
    else:
        print("File not found:", path)
