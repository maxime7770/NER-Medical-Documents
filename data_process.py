import pandas as pd
from difflib import SequenceMatcher




def init_data():
    ''' Rearrange data for further manipulations'''

    df = pd.read_csv('./HummingBird_prototype/data/ddi_description.csv')
    name1 = df['first drug name']
    name2 = df['second drug name']
    sentence = df['description']
    data = []
    for i in range(len(name1)):
            data.append([sentence.iloc[i], [(name1.iloc[i], 'Chemical'), (name2.iloc[i], 'Chemical')]])
    df = pd.DataFrame(data, columns=['text', 'annotation'])
    return df

def matcher(string, pattern):
    ''' Return the start and end index of any pattern present in the text'''

    match_list = []
    pattern = pattern.strip()
    seqMatch = SequenceMatcher(None, string, pattern, autojunk=False)
    match = seqMatch.find_longest_match(0, len(string), 0, len(pattern))
    if (match.size == len(pattern)):
        start = match.a
        end = match.a + match.size
        match_tup = (start, end)
        string = string.replace(pattern, "X" * len(pattern), 1)
        match_list.append(match_tup)
        
    return match_list, string

def mark_sentence(s, match_list):
    ''' Marks all the entities in the sentence as per the BIO scheme. '''

    word_dict = {}
    for word in s.split():
        word_dict[word] = 'O'
        
    for start, end, e_type in match_list:
        temp_str = s[start:end]
        tmp_list = temp_str.split()
        if len(tmp_list) > 1:
            word_dict[tmp_list[0]] = 'B-' + e_type
            for w in tmp_list[1:]:
                word_dict[w] = 'I-' + e_type
        else:
            word_dict[temp_str] = 'B-' + e_type
    return word_dict

def clean(text, clean_punctuation=False, remove_end_point=True):
    ''' Just a helper fuction to add a space before the punctuations for better tokenization '''

    filters = ["!", "#", "$", "%", "&", ".", ":", ";", "<", "=", ">", "?", "@",
               "\\", "_", "`", "{", "}", "~", "'"]

    # cleaning punctation can cause problems with my data        
    if clean_punctuation:
        for i in text:
            if i in filters:
                text = text.replace(i, " " + i)
            
    if remove_end_point:
        return text[:-1]

def create_data(df, filepath):
    ''' The function responsible for the creation of data in the appropriate format '''

    with open(filepath , 'w') as f:
        for text, annotation in zip(df.text, df.annotation):
            text = clean(text)
            text_ = text        
            match_list = []
            for i in annotation:
                a, text_ = matcher(text, i[0])
                match_list.append((a[0][0], a[0][1], i[1]))

            d = mark_sentence(text, match_list)

            for i in d.keys():
                f.writelines(i + ' ' + d[i] +'\n')
            f.writelines('\n')
            
def main():
    ''' main function, combines previous function to create train, dev and test sets'''

    data = init_data()
    data.sample(frac=1).reset_index(drop=True) # shuffle the data

    ## path to save the txt file.
    filepath_train = './HummingBird_prototype/data/train.txt'
    filepath_test = './HummingBird_prototype/data/test.txt'
    filepath_dev = './HummingBird_prototype/data/dev.txt'
    
    ## creating the file.
    length = len(data)
    data_train, data_test, data_dev = data[:int(length*0.8)], data[int(length*0.8):int(length*0.9)], data[int(length*0.9):]
    create_data(data_train, filepath_train)
    create_data(data_dev, filepath_dev)
    create_data(data_test, filepath_test)




if __name__ == '__main__':
    main()
