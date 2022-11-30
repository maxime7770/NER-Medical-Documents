from xml.dom.minidom import CharacterData
from allennlp.predictors.predictor import Predictor
from tqdm import tqdm
from flair.models import SequenceTagger
from flair.data import Sentence
from flair.tokenization import SciSpacyTokenizer
from quantulum3 import parser  # need to install module stemming
import allennlp_models.tagging
import os
import nltk
import re
import benepar
import spacy
import spacy_universal_sentence_encoder


benepar.download('benepar_en3')

nlp = spacy.load('en_core_web_md')       # python3 -m spacy download en_core_web_md
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

nlp_2 = spacy_universal_sentence_encoder.load_model('en_use_lg')

def get_similarity(word1):

    doc_1 = nlp_2(word1)
    dummies = ['people with a disease', 'size', 'height', 'age of the patient', 'patient condition', 'old patient', 'pregnant people'
    ]
    result = 0
    for d in dummies:
        doc_2 = nlp_2(d)
        result = max(result, doc_1.similarity(doc_2))

    return result


class OpenIE:

    def __init__(self):
        self.predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")

    def predict(self, sentence):
        pred = self.predictor.predict(sentence=sentence)
        return pred


class Dosage():

    def __init__(self):
        # self.predictor = SequenceTagger.load("flair/ner-english-ontonotes-fast")
        pass
        

    def predict(self, sentence_):
        results = []
        # sentence = Sentence(sentence_, use_tokenizer=SciSpacyTokenizer())
        # self.predictor.predict(sentence)
        # for annotation_layer in sentence.annotation_layers.keys():
        #     for entity in sentence.get_spans(annotation_layer):
        #         if entity.tag  == 'QUANTITY': 
        #             results.append(entity.text)
        # return results

        quants = parser.parse(sentence_)
        for q in quants:
            print(q.unit.dimensions)
            if q.unit.dimensions[0]['base'] in ['milligram', 'milliliter', 'gram']:
                results.append(q.surface)
        return results





def intersection(li1, s2):
    for word in li1:
        if word in s2:
            return True
    return False



def oie(sentence, model_dosage, model_oie, patient=False, dosage=False):
    ''' 2 pipelines for patient characteristics extraction and dosage extraction '''
    pred = model_oie.predict(sentence)
    if patient:
        doc = nlp(sentence)
        sent = list(doc.sents)[0]
        found = False
        matching_patient = ['patient', 'patients', 'individual', 'individuals', 'people', 'man', 'woman', 'men', 'women', 'male', 'female']
        spans = list(sent._.constituents)
        for x in spans:
            all_parents = []
            if x.text.strip().lower() in matching_patient:
                print('OK ', x)
                parent = x._.parent
                all_parents.append(parent)
                # found_but_no_characteristics = True
            if len(all_parents) > 0:
                print('PARENTS', all_parents)
                parent = min(all_parents, key=lambda x: len(x.text))
                print(list(parent._.children))
                for y in parent._.children:
                    y_label = y._.parse_string.split(' ')[0][1:]
                    if y_label not in ['DT', 'NN', 'NNS', 'IN']: # means you have more than just 'the patient(s)'
                        found = True
                        characteristics = parent.text
                        break

        if found:
            print('SIMILARITY 1: ', get_similarity(characteristics))
            if get_similarity(characteristics) > 0.35:
                return characteristics
            else:
                return None


        found = False
        matching_patient = ['patient', 'patients', 'individual', 'individuals', 'people', 'man', 'woman', 'men', 'women', 'male', 'female']
        matching_characteristics = ['young', 'male', 'female', 'elderly', 'years', 'old', 'with']
        for x in pred['verbs']:
            desc = x['description']
            print(desc)
            args = re.findall('\[ARG.*?\]', desc)
            for arg in args:
                if arg.startswith('[ARG1'):
                    for t in matching_patient:
                        if t in arg:
                            # characteristics = arg
                            # found = True
                            # break
                            if intersection(matching_characteristics, arg):
                                return ' '.join(arg.split(' ')[1:])[:-1]
                            else:
                                for arg_preced in args:
                                    if arg_preced.startswith('[ARG0') or arg_preced.startswith('[ARG2'):
                                        if intersection(matching_characteristics, arg_preced):
                                            characteristics = arg_preced
                                            found= True
                                            break
        if found:
            print(characteristics)
            characteristics = ' '.join(characteristics.split(' ')[1:])[:-1]
            print('SIMILARITY 2: ', get_similarity(characteristics))
            if get_similarity(characteristics) > 0.35:
                return characteristics
        else:
            return None
                            
                        
    if dosage:
        found = False
        matching_text = model_dosage.predict(sentence)
        for x in pred['verbs']:
            desc = x['description']
            args = re.findall('\[ARG.*?\]', desc)
            for arg in args:
                found_drug = False
                if arg.startswith('[ARGM-MNR') or arg.startswith('[ARGM-EXT') or arg.startswith('[ARGM-TMP'):
                    for t in matching_text:
                        if t.replace(' ', '') in arg.replace(' ', ''):
                            found = True
                            for arg_preced in args:
                                if arg_preced.startswith('[ARG0') or arg_preced.startswith('[ARG1') or arg_preced.startswith('[ARG2'):
                                    found_drug = True
                                    drug = arg_preced
                                    break
                            couple = (' '.join(drug.split(' ')[1:])[:-1], t) if found_drug else (None, t)
        return couple if found else (None, None)



        

def pipeline(text, window_size = 2):
    model_dosage = Dosage()
    model_oie = OpenIE()

    results_patient = []
    results_dosage = []

    tokenizer_split_sentences = nltk.data.load('tokenizers/punkt/english.pickle')
    model_url = 'https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz'
    predictor = Predictor.from_path(model_url)

    paragraphs = text.split('\n\n')
    sentences = []
    for p in paragraphs:
        sentences.extend(tokenizer_split_sentences.tokenize(p))
        sentences[-1] = sentences[-1]+ '\n\n'  

    list_coref = [sentences[i] for i in range(window_size)]
    try:
        transformed_chunk = predictor.coref_resolved(' '.join(list_coref))
        paragraphs_transformed = transformed_chunk.split('\n\n')
        sentences_transformed_init = []
        for p in paragraphs_transformed:
            sentences_transformed_init.extend(tokenizer_split_sentences.tokenize(p))
        print('SENTENCES_TRANSFORMED_INIT', sentences_transformed_init)

    except:
        sentences_transformed_init = list_coref
        pass

    all_transformed_sentences = []
    all_sentences = []
    for s in tqdm(range(len(sentences))):

        if s >= window_size:
            sentence = sentences[s]
            list_coref.append(sentence)
            list_coref.pop(0)
            try:
                transformed_chunk = predictor.coref_resolved(' '.join(list_coref))
                paragraphs_transformed = transformed_chunk.split('\n\n')
                for p in paragraphs_transformed:
                    if p != '':
                        sentences_transformed = tokenizer_split_sentences.tokenize(p)
                sentence_transformed = sentences_transformed[-1]
            except:
                sentence_transformed = sentence
                pass

        else:
            try:
                sentence_transformed = sentences_transformed_init[s]
            except:
                print(s)
                print(sentences_transformed_init)
                print(list_coref)
                print(sentences)

                sentence_transformed = sentences_transformed_init[0]       # TO BE CHANGED, JUST TO AVOID ERROR
                

        all_transformed_sentences.append(sentence_transformed)
        all_sentences.append(sentences[s])
        # Open IE

        # For patient information 

        results_patient.append(oie(sentence_transformed, model_dosage, model_oie,  patient=True))

        # For dosage information

        results_dosage.append(oie(sentence_transformed, model_dosage, model_oie, dosage=True))



    return results_patient, results_dosage, all_transformed_sentences, all_sentences




# propagate transformed sentences maybe? (but would also propagate the errors...)


if __name__ == '__main__':
    text = "Aspirin is most of the time an innocuous drug. There are 2 cats in the kitchen. But it can have deleterious effect if it is administered with a dosage of more than 1000 mg. Some adverse effects are exacerbated for some patients. For instance, if they are more than 70 years old, effects will be more pronounced."
    text2 = "Some studies have proven the impact of cytoxin on young female patients. Indeed, it can cause nose bleeding when 500mg or more if taken by the patient. This a test sentence. "
    text3 = "In this paper we try to prove how chloroquine can generate negative effects on elderly people. It can in particular causes stomach pain when the maximum 1000mg dosage is not respected. Finally we show that it has no or little effect on younger patients even when they take more than 1000mg."
    text4 = "Patients with hemoglobin E beta-thalassemia, a severe form of the disease, were found to have impaired hepcidin function and higher TfR1 levels as a result of an increased erythropoietic drive stemming from the continuously failing erythropoiesis that is caused by improper hemoglobin production [56]."
    text5 = "Another study observed a 3-year disease free survival rate of 80 percent, and an overall survival rate of 82 percent in cervical cancer patients."
    text6 = "A study reported that individuals with hereditary hemochromatosis exhibit an increased risk for developing cancer, particularly in the liver and primarily hepatocellular carcinoma as opposed to biliary tract related cancers."
    text7 = "Due to oxygen's atomic nature, its reduction must proceed in a stepwise fashion of individual electron additions and reactive intermediates."
    text8 = "Chloroquine analogues have also been found to have metabolic, antithrombotic, antineoplastic, and antiviral effects, and have been hypothesized as targeted agents against coronavirus infection since the 2003 SARS outbreak [25,26]."
    results_patient, result_dosage, all_transformed_sentences, all_sentences = pipeline(text + '\n' + text2 + '\n' + text3 + '\n' + text6 + '\n' + text7 + '\n' + text8, window_size=3)

    for i in range(len(all_transformed_sentences)):
        print('\n \n')
        print('Original sentence: ', all_sentences[i].strip())
        print('\n')
        print('Transformed sentence: ', all_transformed_sentences[i].strip())
        print('\n')
        print('Detected patients: ', results_patient[i])
        print('\n')
        print('Detected dosages: ', result_dosage[i])
        print('\n')
        print('=============================================')

        
    # print(get_similarity('for elderly people'))
    # print(get_similarity('patients more than 70 years old'))
    # print(get_similarity('individual electrons'))


