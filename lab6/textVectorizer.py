
import pandas as pd
import numpy as np
import sys
import os
from nltk.stem import PorterStemmer

def create_ground_truth(path, out):
    f = open(f'{out}/ground_truth.csv', 'w')
    f.write('file,author,size\n')
    for d in ['C50train', 'C50test']:
        new_path = path + 'C50/' + d
        authors = os.listdir(new_path)
        for author in authors:
           if author != '.DS_Store':
               for text in os.listdir(new_path + '/' + author):
                   if text != '.ipynb_checkpoints':
                       size = os.path.getsize(new_path + '/' + author + '/' + text)
                       f.write(f'{text},{author},{size}\n')
    f.close()

def get_stopwords():
    stopwords_long = ['a', 'able', 'about', 'above', 'abst', 'accordance', 'according',
       'accordingly', 'across', 'act', 'actually', 'added', 'adj',
       'adopted', 'affected', 'affecting', 'affects', 'after',
       'afterwards', 'again', 'against', 'ah', 'all', 'almost', 'alone',
       'along', 'already', 'also', 'although', 'always', 'am', 'among',
       'amongst', 'an', 'and', 'announce', 'another', 'any', 'anybody',
       'anyhow', 'anymore', 'anyone', 'anything', 'anyway', 'anyways',
       'anywhere', 'apparently', 'approximately', 'are', 'aren', 'arent',
       'arise', 'around', 'as', 'aside', 'ask', 'asking', 'at', 'auth',
       'available', 'away', 'awfully', 'b', 'back', 'be', 'became',
       'because', 'become', 'becomes', 'becoming', 'been', 'before',
       'beforehand', 'begin', 'beginning', 'beginnings', 'begins',
       'behind', 'being', 'believe', 'below', 'beside', 'besides',
       'between', 'beyond', 'biol', 'both', 'brief', 'briefly', 'but',
       'by', 'c', 'ca', 'came', 'can', 'cannot', "can't", 'cause',
       'causes', 'certain', 'certainly', 'co', 'com', 'come', 'comes',
       'contain', 'containing', 'contains', 'could', 'couldnt', 'd',
       'date', 'did', "didn't", 'different', 'do', 'does', "doesn't",
       'doing', 'done', "don't", 'down', 'downwards', 'due', 'during',
       'e', 'each', 'ed', 'edu', 'effect', 'eg', 'eight', 'eighty',
       'either', 'else', 'elsewhere', 'end', 'ending', 'enough',
       'especially', 'et', 'et-al', 'etc', 'even', 'ever', 'every',
       'everybody', 'everyone', 'everything', 'everywhere', 'ex',
       'except', 'f', 'far', 'few', 'ff', 'fifth', 'first', 'five', 'fix',
       'followed', 'following', 'follows', 'for', 'former', 'formerly',
       'forth', 'found', 'four', 'from', 'further', 'furthermore', 'g',
       'gave', 'get', 'gets', 'getting', 'give', 'given', 'gives',
       'giving', 'go', 'goes', 'gone', 'got', 'gotten', 'h', 'had',
       'happens', 'hardly', 'has', "hasn't", 'have', "haven't", 'having',
       'he', 'hed', 'hence', 'her', 'here', 'hereafter', 'hereby',
       'herein', 'heres', 'hereupon', 'hers', 'herself', 'hes', 'hi',
       'hid', 'him', 'himself', 'his', 'hither', 'home', 'how', 'howbeit',
       'however', 'hundred', 'i', 'id', 'ie', 'if', "i'll", 'im',
       'immediate', 'immediately', 'importance', 'important', 'in', 'inc',
       'indeed', 'index', 'information', 'instead', 'into', 'invention',
       'inward', 'is', "isn't", 'it', 'itd', "it'll", 'its', 'itself',
       "i've", 'j', 'just', 'k', 'keep', 'keeps', 'kept', 'keys', 'kg',
       'km', 'know', 'known', 'knows', 'l', 'largely', 'last', 'lately',
       'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let',
       'lets', 'like', 'liked', 'likely', 'line', 'little', "'ll", 'look',
       'looking', 'looks', 'ltd', 'm', 'made', 'mainly', 'make', 'makes',
       'many', 'may', 'maybe', 'me', 'mean', 'means', 'meantime',
       'meanwhile', 'merely', 'mg', 'might', 'million', 'miss', 'ml',
       'more', 'moreover', 'most', 'mostly', 'mr', 'mrs', 'much', 'mug',
       'must', 'my', 'myself', 'n', 'na', 'name', 'namely', 'nay', 'nd',
       'near', 'nearly', 'necessarily', 'necessary', 'need', 'needs',
       'neither', 'never', 'nevertheless', 'new', 'next', 'nine',
       'ninety', 'no', 'nobody', 'non', 'none', 'nonetheless', 'noone',
       'nor', 'normally', 'nos', 'not', 'noted', 'nothing', 'now',
       'nowhere', 'o', 'obtain', 'obtained', 'obviously', 'of', 'off',
       'often', 'oh', 'ok', 'okay', 'old', 'omitted', 'on', 'once', 'one',
       'ones', 'only', 'onto', 'or', 'ord', 'other', 'others',
       'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside',
       'over', 'overall', 'owing', 'own', 'p', 'page', 'pages', 'part',
       'particular', 'particularly', 'past', 'per', 'perhaps', 'placed',
       'please', 'plus', 'poorly', 'possible', 'possibly', 'potentially',
       'pp', 'predominantly', 'present', 'previously', 'primarily',
       'probably', 'promptly', 'proud', 'provides', 'put', 'q', 'que',
       'quickly', 'quite', 'qv', 'r', 'ran', 'rather', 'rd', 're',
       'readily', 'really', 'recent', 'recently', 'ref', 'refs',
       'regarding', 'regardless', 'regards', 'related', 'relatively',
       'research', 'respectively', 'resulted', 'resulting', 'results',
       'right', 'run', 's', 'said', 'same', 'saw', 'say', 'saying',
       'says', 'sec', 'section', 'see', 'seeing', 'seem', 'seemed',
       'seeming', 'seems', 'seen', 'self', 'selves', 'sent', 'seven',
       'several', 'shall', 'she', 'shed', "she'll", 'shes', 'should',
       "shouldn't", 'show', 'showed', 'shown', 'showns', 'shows',
       'significant', 'significantly', 'similar', 'similarly', 'since',
       'six', 'slightly', 'so', 'some', 'somebody', 'somehow', 'someone',
       'somethan', 'something', 'sometime', 'sometimes', 'somewhat',
       'somewhere', 'soon', 'sorry', 'specifically', 'specified',
       'specify', 'specifying', 'state', 'states', 'still', 'stop',
       'strongly', 'sub', 'substantially', 'successfully', 'such',
       'sufficiently', 'suggest', 'sup', 'sure', 't', 'take', 'taken',
       'taking', 'tell', 'tends', 'th', 'than', 'thank', 'thanks',
       'thanx', 'that', "that'll", 'thats', "that've", 'the', 'their',
       'theirs', 'them', 'themselves', 'then', 'thence', 'there',
       'thereafter', 'thereby', 'thered', 'therefore', 'therein',
       "there'll", 'thereof', 'therere', 'theres', 'thereto', 'thereupon',
       "there've", 'these', 'they', 'theyd', "they'll", 'theyre',
       "they've", 'think', 'this', 'those', 'thou', 'though', 'thoughh',
       'thousand', 'throug', 'through', 'throughout', 'thru', 'thus',
       'til', 'tip', 'to', 'together', 'too', 'took', 'toward', 'towards',
       'tried', 'tries', 'truly', 'try', 'trying', 'ts', 'twice', 'two',
       'u', 'un', 'under', 'unfortunately', 'unless', 'unlike',
       'unlikely', 'until', 'unto', 'up', 'upon', 'ups', 'us', 'use',
       'used', 'useful', 'usefully', 'usefulness', 'uses', 'using',
       'usually', 'v', 'value', 'various', "'ve", 'very', 'via', 'viz',
       'vol', 'vols', 'vs', 'w', 'want', 'wants', 'was', "wasn't", 'way',
       'we', 'wed', 'welcome', "we'll", 'went', 'were', "weren't",
       "we've", 'what', 'whatever', "what'll", 'whats', 'when', 'whence',
       'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein',
       'wheres', 'whereupon', 'wherever', 'whether', 'which', 'while',
       'whim', 'whither', 'who', 'whod', 'whoever', 'whole', "who'll",
       'whom', 'whomever', 'whos', 'whose', 'why', 'widely', 'willing',
       'wish', 'with', 'within', 'without', "won't", 'words', 'world',
       'would', "wouldn't", 'www', 'x', 'y', 'yes', 'yet', 'you', 'youd',
       "you'll", 'your', 'youre', 'yours', 'yourself', 'yourselves',
       "you've", 'z', 'zero', '']
    ps = PorterStemmer()
    stopwords_final = list(set([ps.stem(''.join(stopword.split("'"))) for stopword in stopwords_long]))
    return stopwords_final


def process_text(text):
   to_replace_no_space = [':', '.', ',', '$', '%', '!',
                          "\'", '"', '(', ')', '=', '?', '+',
                          '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
   to_replace_yes_space = ['\n', '\t', '-']
   temp = text.lower()
   for c in to_replace_no_space:
      temp = temp.replace(c, '')
   for c in to_replace_yes_space:
      temp = temp.replace(c, ' ')
   return temp.split(' ')


def get_vocab(df, path, stopwords):
   ps = PorterStemmer()
   vocab = set([])

   for i in range(df.shape[0]):

      file = df.iloc[i, :]['file']
      author = df.iloc[i, :]['author']

      if i < 2500:
         path_new = path + 'C50/C50train/' + author + '/' + file
      else:
         path_new = path + 'C50/C50test/' + author + '/' + file

      f = open(path_new)

      words = process_text(f.read())
      words = set([ps.stem(word) for word in words])

      vocab.update(words)

   vocab_final = list(vocab - set(stopwords))

   return vocab_final


def create_files(df, vocab, path, out_path):
   df_tf = pd.DataFrame(columns=vocab)
   ps = PorterStemmer()

   for i in range(df.shape[0]):

      file = df.iloc[i, :]['file']
      author = df.iloc[i, :]['author']

      if i < 2500:
         path_new = path + 'C50/C50train/' + author + '/' + file
      else:
         path_new = path + 'C50/C50test/' + author + '/' + file

      vocab_dict = {}

      for word in vocab:
         vocab_dict[word] = 0

      f = open(path_new)

      words = process_text(f.read())
      words = set([ps.stem(word) for word in words])

      for word in words:
         try:
            vocab_dict[word] += 1
         except:
            None

      obs = pd.DataFrame([list(vocab_dict.values())], columns=vocab, index=[i])
      df_tf = pd.concat([df_tf, obs], axis=0)

   df_tf.to_csv(f'{out_path}term_freqs.csv', index=False)


def write_other_files(tf_file, out_path):
   df_tf = pd.read_csv(tf_file)
   df = df_tf.sum()

   n = df_tf.shape[0]
   idf = np.log2(n / df)
   df_tf_idf = df_tf * idf

   df_tf_idf.to_csv(f'{out_path}tf_idf.csv', index=False)
   pd.DataFrame(df).reset_index().rename(columns={'index': 'term', 0: 'idf'}).to_csv(f'{out_path}doc_freqs.csv',
                                                                                     index=False)


def main():
    args = sys.argv

    path = args[1]
    out = args[2]

    create_ground_truth(path, out)

    df_gt = pd.read_csv(f'{out}ground_truth.csv')

    stopwords = get_stopwords()

    vocab = get_vocab(df_gt, path, stopwords)

    create_files(df_gt, vocab, path, out)

    del vocab
    del stopwords
    del df_gt

    write_other_files(f'{out}term_freqs.csv', out)
    

if __name__ == '__main__':
    main()