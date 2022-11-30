import streamlit
from ocr import extract_images_from_pdf, ocr_core
from evaluation import higlight
import os
import pdfplumber
import base64


def process_file(file, choices=[True, False, False, False, False]):
    ''' Takes as input a pdf file, store the file and run the pipeline '''

    dir_name = file.name[:-4]
    if not os.path.exists("./HummingBird_prototype/processed_files/"+dir_name):
        os.mkdir("./HummingBird_prototype/processed_files/"+dir_name)
        with open(os.path.join("./HummingBird_prototype/processed_files/"+dir_name,file.name),"wb") as f:
            f.write(file.getbuffer())
    extract_images_from_pdf(file.name)
    if file.name.endswith('.pdf'):
        ocr_core(file.name)
        higlight(file.name, choices)
    else:
        file_name = file.name[:-4] + ".pdf"
        ocr_core(file_name)
        higlight(file_name, choices)

    

def display_pdf(file: str):
    ''' Display the pdf file from its path '''

    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="650" height="1000" type="application/pdf">'

    # Displaying File
    streamlit.markdown(pdf_display, unsafe_allow_html=True)
  


def run():
  ''' Main function to run the app '''

  # set the title
  streamlit.title('HummingBird Prototype')

  # add a file explorer to load multiple files
  uploaded_files = streamlit.file_uploader('Upload your file here', type=['pdf', 'png', 'jpg'], accept_multiple_files=True)
  
  # create a bool static variable to save if the user want to display the full transcript
  streamlit.session_state["view_file"] = False

  # bool static variable: the value is True once the file has been processed and the results are ready
  streamlit.session_state["view_results"] = False

  codes = {'Chemicals': '#FFFF00', 'Diseases': '#00FF00', 'Dates': '#00B3FF', 'Adverse effects': '#FF0000', 'Doses': '#FF00FF'}

  # add a slectbox to select the file to process
  option = streamlit.selectbox(
    'Select the file you want to process',
    tuple((uploaded_file.name for uploaded_file in uploaded_files)))

  # add a selectbox to select the type of entities to extract
  with streamlit.expander(label='Select the entities you want to higlight', expanded=False):
        chemicals = streamlit.checkbox('Chemicals', value=True)
        diseases = streamlit.checkbox('Diseases', value=False)
        dates = streamlit.checkbox('Dates', value=False)
        adverse = streamlit.checkbox('Adverse effects', value=False)
        doses = streamlit.checkbox('Doses', value=False)

  choices = [chemicals, diseases, dates, adverse, doses]

  # add a button to process the selected file
  if streamlit.button('Process file') and option is not None:
      uploaded_file = [uploaded_file for uploaded_file in uploaded_files if uploaded_file.name == option][0]
      process_file(uploaded_file, choices)
      streamlit.session_state["view_results"] = True


  if streamlit.button('View results') and option is not None:
        col1, col2 = streamlit.columns([10000, 1], gap='large')
        with col1:
            uploaded_file = [uploaded_file for uploaded_file in uploaded_files if uploaded_file.name == option][0]
            file_name = uploaded_file.name[:-4]
            if not os.path.exists("./HummingBird_prototype/processed_files/"+file_name+"/"+file_name+"_highlighted.pdf"):
                streamlit.write("The file has not been processed yet")
            else:
                file_name = uploaded_file.name
                display_pdf("./HummingBird_prototype/processed_files/"+file_name[:-4]+"/"+file_name[:-4]+"_highlighted.pdf")
        with col2:
            i = 0
            for key, value in codes.items():
                if choices[i]:
                    streamlit.color_picker(key, value, disabled=False)
                i += 1

  # add a button to toggle or untoggle the full transcript
  if streamlit.button('View file') and option is not None:
      streamlit.session_state["view_file"] = not streamlit.session_state["view_file"]

      
  if streamlit.session_state["view_file"]:
      uploaded_file = [uploaded_file for uploaded_file in uploaded_files if uploaded_file.name == option][0]
      if uploaded_file.name.endswith('.pdf'):
          base64_pdf = base64.b64encode(uploaded_file.read()).decode('utf-8')
          pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="600" height="1300" type="application/pdf">' 
          streamlit.markdown(pdf_display, unsafe_allow_html=True)
      elif uploaded_file.name.endswith('.png') or uploaded_file.name.endswith('.jpg'):
          streamlit.image(uploaded_file, use_column_width=True)


  if streamlit.button('View transcript') and option is not None:
      uploaded_file = [uploaded_file for uploaded_file in uploaded_files if uploaded_file.name == option][0]
      try:
        with pdfplumber.open(uploaded_file) as pdf:
          for page in pdf.pages:
            streamlit.write(page.extract_text())
      except:
        streamlit.write("None")


if __name__ == '__main__':
    run()
