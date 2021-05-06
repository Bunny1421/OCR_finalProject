import cv2 
import pytesseract
import matplotlib.pyplot as plt
from pytesseract import Output
import numpy as np

from PIL import ImageFont, ImageDraw, Image

img = cv2.imread('Capture.PNG')

pil_im = Image.fromarray(img)
draw = ImageDraw.Draw(pil_im)
# Adding custom options
custom_config = r'--oem 3 --psm 6'
# pytesseract.image_to_string(img, 'tha')
print(pytesseract.image_to_string(img, 'tha+eng', config=custom_config))
results = pytesseract.image_to_data(img,'tha+eng',output_type=Output.DICT)
print(results)
font_futura = ImageFont.truetype("TH Krub.ttf", 20)

for i in range(0, len(results["text"])):
    # extract the bounding box coordinates of the text region from
    # the current result
    x = results["left"][i]
    y = results["top"][i]
    w = results["width"][i]
    h = results["height"][i]
    # extract the OCR text itself along with the confidence of the
    # text localization
    text = results["text"][i]
    print(text)
    conf = int(results["conf"][i])
    if conf >60:
        # display the confidence and text to our terminal
        print("Confidence: {}".format(conf))
        print("Text: {}".format(text))
        print('1 ',text)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        draw.text((20, 75), text, font=font_futura,fill=(0,0,0,255))
        # cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX,1.2, (255, 0, 255), 3)
        # cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
# show the output image
cv2.imshow("Image", img)
cv2.waitKey(0)
