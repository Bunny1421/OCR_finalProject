import easyocr

imgPath = './imgs/output/sample6.jpg'

reader = easyocr.Reader(['th','en'], gpu=False)
result = reader.readtext(imgPath, detail=0)
print(result)

# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\unif\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
# imgPath = './imgs/output/sample6.jpg'
# text = pytesseract.image_to_string(imgPath, lang='tha+eng')
# print(text)