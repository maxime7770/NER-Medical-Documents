import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import os 
from tqdm import tqdm
import streamlit




def extract_images_from_pdf(filename: str):
    ''' Takes as input a pdf file (its name) and convert each page into an image'''
    
    root = './HummingBird_prototype/processed_files/' + filename[:-4] + '/'

    # if the file is a pdf, convert it into images
    if filename.endswith('.pdf'):
        streamlit.write("Extracting images from pdf...")
        convert_from_path(root+filename, output_folder=root, fmt='png', output_file=filename[:1]+'page')
        streamlit.write("Extraction done")

    # if not a pdf, save a pdf copy (it will be used for highlighting later [UPDATE: higlighting does not work on images converted to pdf])
    elif filename.endswith('.png') or filename.endswith('.jpg'):
        image = Image.open(root+filename)
        image.convert('RGB').save(root+filename[:-4]+'.pdf')



def ocr_core(filename: str):
    ''' Takes as input a pdf file (its name) and run the OCR on each image created in
    the previous function
    '''

    streamlit.write("Performing OCR...")
    root = './HummingBird_prototype/processed_files/' + filename[:-4] + '/'
    for file in tqdm(os.listdir(root)):
        if file.endswith('.png') or file.endswith('.jpg'):
            result = pytesseract.image_to_string(Image.open(root+file))
            with open(root+file[:-4]+'-ocr.txt', 'w') as f:
                f.write(result)
    streamlit.write("OCR done")


if __name__ == '__main__':
    ocr_core('1.pdf')
