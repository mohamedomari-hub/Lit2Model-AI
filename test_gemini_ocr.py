from src.gemini_ocr import extract_equations_with_gemini

PDF_PATH = "data/sample_papers/MetRep_Dexamethasone_Adm.pdf"

result = extract_equations_with_gemini(PDF_PATH)

print(result)