from flair.data import Sentence
from flair.models import SequenceTagger, TextClassifier
from flair.tokenization import SciSpacyTokenizer
from transformers import pipeline, TextClassificationPipeline, AutoTokenizer, TFBertForTokenClassification, BertForSequenceClassification, AutoModelForSequenceClassification
from transformers.trainer import Trainer, TrainingArguments
from stqdm import stqdm
from allennlp.predictors.predictor import Predictor
import os
import fitz
import streamlit
import wikipedia
import nltk.data
import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class InferenceADE:  
    ''' Voting classifier using 3 different models for ADE detection '''

    def __init__(self, pipeline_scibert, pipeline_biolink, model_hunflair):
        self.f1_score_biolink = 0.96   # not real f1 scores for now
        self.f1_score_scibert = 0.80
        self.f1_score_hunflair = 0.90
        self.pipeline_scibert = pipeline_scibert
        self.pipeline_biolink = pipeline_biolink
        self.model_hunflair = model_hunflair

    def __call__(self, sentence):

        result_bert = self.pipeline_scibert(sentence)[0]
        result_biolink = self.pipeline_biolink(sentence)[0]
        s = Sentence(sentence)
        self.model_hunflair.predict(s)
        result_hunflair = s.labels[0].to_dict()

        if result_bert['label'] == 'LABEL_0':
            pred_scibert = [result_bert['score'], 1-result_bert['score']]
        elif result_bert['label'] == 'LABEL_1':
            pred_scibert = [1-result_bert['score'], result_bert['score']]

        if result_biolink['label'] == 'LABEL_0':
            pred_biolink = [result_biolink['score'], 1-result_biolink['score']]
        elif result_biolink['label'] == 'LABEL_1':
            pred_biolink = [1-result_biolink['score'], result_biolink['score']]

        if result_hunflair['value'] == '0':
            pred_hunflair = [result_hunflair['confidence'], 1-result_hunflair['confidence']]
        elif result_hunflair['value'] == '1':
            pred_hunflair = [1-result_hunflair['confidence'], result_hunflair['confidence']]

        # voting classifier

        weighted_average_1 = float((self.f1_score_biolink * pred_biolink[0] + self.f1_score_scibert * pred_scibert[0] + self.f1_score_hunflair * pred_hunflair[0]) / (self.f1_score_biolink + self.f1_score_scibert + self.f1_score_hunflair))
        weighted_average_2 = float((self.f1_score_biolink * pred_biolink[1] + self.f1_score_scibert * pred_scibert[1] + self.f1_score_hunflair * pred_hunflair[1]) / (self.f1_score_biolink + self.f1_score_scibert + self.f1_score_hunflair))

        return [weighted_average_1, weighted_average_2]



def extraction(filename: str, choices: list[bool], use_streamlit: bool = True):
    ''' Takes as input the name of a pdf file and extract the wanted entities from it 
        Outputs a dictionary with the entities' names as keys and the list of entities as values
    '''
    tokenizer_split_sentences = nltk.data.load('tokenizers/punkt/english.pickle')
    root = './NER-Medical-Document/processed_files/' + filename[:-4] + '/'
    results = {}
    models = {}
    limit = 20 # maximum number of files to process, used for testing if the pdf file is too big
    ind = 0
    # only add the necessary models to the dictionary 'models' (avoid unnecessary loading)
    model_url = 'https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz'
    predictor = Predictor.from_path(model_url)

    if choices[0]:
        tagger_chemicals: SequenceTagger = SequenceTagger.load('./NER-Medical-Document/training_results/best-model.pt')
        models[0] = tagger_chemicals

    if choices[1]:
        tagger_diseases: SequenceTagger = SequenceTagger.load('hunflair-disease')
        models[1] = tagger_diseases

    if choices[2]:
        tagger_dates = SequenceTagger.load("flair/ner-english-ontonotes-fast")
        models[2] = tagger_dates

    if choices[3]:
        # 3 different methods for ADE detection
        method = 2

        if method == 1:
            # Use a token classification model (classify a token as an ADE, not a sentence)
            model_adverse_name = "abhibisht89/spanbert-large-cased-finetuned-ade_corpus_v2" # model name from huggingface.co/models
            model_adverse = TFBertForTokenClassification.from_pretrained(model_adverse_name, from_pt=True)
            tokenizer_adverse = AutoTokenizer.from_pretrained(model_adverse_name)
            models[3] = pipeline("token-classification", model = model_adverse, tokenizer = tokenizer_adverse, grouped_entities=True)

        elif method == 2:
            # Sentence classification: use HunFlair model + negation detection
            tokenizer_neg = AutoTokenizer.from_pretrained("bvanaken/clinical-assertion-negation-bert")
            model_neg = AutoModelForSequenceClassification.from_pretrained("bvanaken/clinical-assertion-negation-bert")
            pipeline_neg = TextClassificationPipeline(model=model_neg, tokenizer=tokenizer_neg)
            model_hunflair = TextClassifier.load('./NER-Medical-Document/training_results/flair_bert/best-model.pt')
            models[3] = model_hunflair

        elif method == 3:
            # Sentence classification: use the InferenceADE class to design a voting classifier
            model_scibert_name = 'NER-Medical-Document/training_results/scibert_scivocab_uncased'
            model_scibert = BertForSequenceClassification.from_pretrained(model_scibert_name)
            tokenizer_scibert = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            pipeline_scibert = pipeline("text-classification", model = model_scibert, tokenizer = tokenizer_scibert)

            model_biolink_name = 'NER_Medical-Document/training_results/BioLinkBERT-base'
            model_biolink = BertForSequenceClassification.from_pretrained(model_biolink_name)
            tokenizer_biolink = AutoTokenizer.from_pretrained('michiyasunaga/BioLinkBERT-base')
            pipeline_biolink = pipeline("text-classification", model = model_biolink, tokenizer = tokenizer_biolink)

            model_hunflair = TextClassifier.load('./NER-Medical-Document/training_results/flair_bert/best-model.pt')

            models[3] = InferenceADE(pipeline_scibert, pipeline_biolink, model_hunflair)

    if choices[4]:
        tagger_doses = SequenceTagger.load("flair/ner-english-ontonotes-fast")
        models[4] = tagger_doses


    dic = {0: 'Chemicals', 1: 'Diseases', 2: 'Dates', 3: 'Adverse effects', 4: 'Doses'}

    info = 'Extracting '
    for j, c in enumerate(choices):
        if c:
            info += dic[j].lower() + ' and '
    streamlit.write(info[:-5] + ' entities...')

    local_results_chemicals = []
    local_results_diseases = []
    local_results_dates = []
    local_results_adverse = []
    local_results_doses = []
    dic_doses_chemicals = {}

    for i, file in enumerate(os.listdir(root)):
        if file.endswith('.txt') and ind < limit:
            ind += 1
            with open(root+file, 'r') as f:
                paragraphs = f.read().split('\n\n')
                sentences = []
                for p in paragraphs:
                    sentences.extend(tokenizer_split_sentences.tokenize(p))
                    sentences[-1] = sentences[-1]+ '\n\n'                


                if not use_streamlit:
                    size_window = 3
                    list_coref = [sentences[i] for i in range(size_window)]
                    try:
                        prediction = predictor.predict(document=' '.join(list_coref))
                    except:
                        pass

                for s in stqdm(range(len(sentences))):
                    sentence_ = sentences[s]

                    ### Coreference resolution ###
                    if not use_streamlit:
                        if s >= size_window:
                            list_coref.append(sentence_)
                            list_coref.pop(0)
                            try:
                                prediction = predictor.predict(document=' '.join(list_coref))
                                transformed_chunk = predictor.coref_resolved(' '.join(list_coref))
                                paragraphs2 = transformed_chunk.split('\n\n')
                                for p in paragraphs2:
                                    sentences2 = tokenizer_split_sentences.tokenize(p)
                                sentence_transformed = sentences2[-1]
                            except:
                                sentence_transformed = sentence_
                                pass
                        else:
                            sentence_transformed = sentence_

                    sentence_ = sentence_.replace('\n', ' ')
                    if len(sentence_) >= 4:
                        print(sentence_)
                        sentence = Sentence(sentence_, use_tokenizer=SciSpacyTokenizer())
                        for j, c in enumerate(choices):
                            if c:
                                tagger = models[j]
                                if dic[j] == 'Adverse effects':
                                    if method == 1:
                                        result = tagger(sentence_)
                                        if result != []:
                                            for entity in tagger(sentence_):
                                                if entity['entity_group'] == 'ADR':
                                                    local_results_dates.append(entity['word'])
                                    elif method == 2:
                                        sentence = Sentence(sentence_, use_tokenizer=SciSpacyTokenizer())  # create new instance of Sentence
                                        tagger.predict(sentence)
                                        result = sentence.labels[0].to_dict()
                                        if result['value'] == '1':
                                            if not pipeline_neg(sentence_)[0]['label'] == 'ABSENT':
                                                print('DETECTED')
                                                local_results_adverse.append(sentence_)
                                    elif method == 3:
                                        result = tagger(sentence_)
                                        print(result)
                                        if result[1] > 0.5:
                                            local_results_adverse.append(sentence_)
                                        else:
                                            models[0].predict(sentence)
                                            found = False
                                            for annotation_layer in sentence.annotation_layers.keys():
                                                for entity in sentence.get_spans(annotation_layer):
                                                        found = True
                                                        sentence_2 = sentence_.replace(entity.text, 'aspirin')
                                            if found: 
                                                result = tagger(sentence_2)
                                                print(result)
                                                if result[1] > 0.5:
                                                    local_results_adverse.append(sentence_)
                                else: 
                                    tagger.predict(sentence)
                                    for annotation_layer in sentence.annotation_layers.keys():
                                        for entity in sentence.get_spans(annotation_layer):
                                            if dic[j] == 'Chemicals':
                                                local_results_chemicals.append(entity.text)
                                                detected_chemicals = True
                                                entity_chemical = entity.text
                                            elif dic[j] == 'Diseases':
                                                local_results_diseases.append(entity.text)
                                            elif dic[j] == 'Dates':
                                                if entity.tag == 'DATE':
                                                    local_results_dates.append(entity.text)
                                            elif dic[j] == 'Doses':
                                                if entity.tag == 'QUANTITY':
                                                    local_results_doses.append(entity.text)
                                                    if detected_chemicals:
                                                        print('YES')
                                                        print(detected_chemicals)
                                                        print(sentence)
                                                        dic_doses_chemicals[entity.text] = entity_chemical
                                                    else:
                                                        dic_doses_chemicals[entity.text] = 'unknown'
                        detected_chemicals = False
    for j, c in enumerate(choices):
        if c:
            if dic[j] == 'Chemicals':
                # next line is to avoid detecting some characters as 'drugs' (happened sometimes)
                local_results_chemicals = [x for x in local_results_chemicals if x not in ['(', ')', '[', ']', '{', '}', ' ', '']]
                results[dic[j]] = list(set(local_results_chemicals))
            elif dic[j] == 'Diseases':
                results[dic[j]] = list(set(local_results_diseases))
            elif dic[j] == 'Dates':
                results[dic[j]] = list(set(local_results_dates))
            elif dic[j] == 'Adverse effects':
                results[dic[j]] = list(set(local_results_adverse))
            elif dic[j] == 'Doses':
                results[dic[j]] = list(set(local_results_doses))


    streamlit.write('Done!')
    return results



def higlight(filename: str, choices: list[bool]):
    ''' Highlight the entities chosen in the pdf file whose name is 'filename' '''

    root = './NER-Medical-Document/processed_files/' + filename[:-4] + '/'
    pdf_input = fitz.open(root+filename)
    results = extraction(filename, choices)
    streamlit.write('Highlighting entities...')
    text_instances = {}

    # go through all the pages of the chosen pdf file
    for page in pdf_input:

        # search all the occurences of the entities in the page
        for name, entities in results.items():
            text_instances[name] = [page.search_for(text) for text in entities]

        for name, instances in text_instances.items():
            add_definition = False
            if name == 'Chemicals':
                color = (1, 1, 0)
                add_definition = True
            elif name == 'Diseases':
                color = (0, 1, 0)
                add_definition = True
            elif name == 'Dates':
                color = (0, 0.7, 1)
                add_definition = False
            elif name == 'Adverse effects':
                color = (1, 0, 0)
                add_definition = False
            elif name == 'Doses':
                color = (1, 0, 1)
                add_definition = False

            # highlight each occurence of the entity in the page
            for i, inst in enumerate(instances):
                for x in inst:
                    # handle the case where an entity should not be highlighted (see README): the idea is too check the surrounding characters to 
                    # detect if the occurence of the entity is part of another word or not

                    # check the typical distance between 2 letters in the word (because it depends on the font size)
                    dist_letters = (x[2]-x[0])/len(results[name][i])
                    # draw a larger rectangle to check the surrounding characters
                    rect_larger = fitz.Rect(x[0]-dist_letters, x[1], x[2]+dist_letters, x[3])
                    non_accepted_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
                    word = page.get_textbox(rect_larger).lower()
                    for sub in results[name][i].split():
                        word = word.replace(sub.lower(), '')
                    keep = True
                    for l in word:
                        if l in non_accepted_chars:
                            keep = False
                    if not keep:
                        continue     # ignore the occurence of the entity if it is part of another word

                    annot = page.add_highlight_annot(x)
                    annot.set_colors({"stroke": color})
                    annot.set_opacity(0.4)
                    if add_definition:
                        try:
                            annot.set_popup(x)
                            info = annot.info
                            info["title"] = "Definition"
                            if name == 'Chemicals':
                                info["content"] = wikipedia.summary(results[name][i] +  ' (drug)').split('.')[0]
                            else:
                                info["content"] = wikipedia.summary(results[name][i] + f' ({name.lower()[:-1]})').split('.')[0]
                            annot.set_info(info)
                        except:
                            pass
                    annot.update()
                
    if os.path.exists(root+filename[:-4]+'_highlighted.pdf'):
            os.remove(root+filename[:-4]+'_highlighted.pdf')
    pdf_input.save(root+filename[:-4]+'_highlighted.pdf')
    streamlit.write('Done!')





if __name__ == '__main__':
    print(extraction('0.txt', [True, True, True, True, True], use_streamlit=False))
