import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import *
from scipy.sparse import csr_matrix, find
import string
from os import remove
import warnings as wrn

class myMNB:
    # CONSTRUCTOR
    def __init__(self, languages=None):
        self.classes = languages
        self.priors = {}
        self.log_densities = {}
        self.size_voc = 0

    # GETTERS
    def get_classes(self):
        return self.classes

    def get_priors(self, label):
        return self.priors[label]

    def get_log_densities(self, label):
        return self.log_densities[label]

    # SETTERS
    def set_classes(self, classes):
        self.classes = classes

    def set_priors(self, label, value):
        self.priors[label] = value

    def set_log_densities(self, label, densities):
        self.log_densities[label] = densities

    # METHODS

    def fit(self, x_train, y_train):
        # x_train is the sparsified data
        n_samples, self.size_voc = x_train.shape
        if self.get_classes() == None:
            self.set_classes( np.unique(y_train) ) # setting the classes
        y_train = np.array(y_train)
        for index_l, label in enumerate(self.get_classes()):
            x_train_l = x_train[ y_train == label ]  # gets the training set having label l
            pi_hat = x_train_l.shape[0] / n_samples
            self.set_priors(str(label), pi_hat)
            n_l_j = np.zeros(self.size_voc, dtype=np.float64 )
            W_l = 0
            for word_i in range(self.size_voc):
                n_l_j[word_i] = np.sum(x_train_l[:, word_i])
                    # sum of the occurrences of the j-th word in x_train_l
                W_l += n_l_j[word_i]
                    # progressively computes the total number of words occurrences in the l-th class
            log_f_hat_l_i = np.log(n_l_j + 1) - np.log(W_l + self.size_voc)
                # array-wise smoothed estimated of the conditional densities
            self.set_log_densities(label, log_f_hat_l_i)

    def calc_posteriors(self, x_new):
        # computation of the posterior probabilities
        if len(self.get_classes()) == 0:
            raise ValueError('The model has not been fitted yet!')
        x_de_sparse = find(x_new) # to access the elements stored in the sparse matrix
        x_attr = x_de_sparse[1] # gets the indexes of the words of x_new in the vocabulary
        x_new = x_de_sparse[2] # gets the occurrences of words in x_new
        log_posterior = np.zeros(len(self.get_classes()), dtype=np.float64)
        for index_l, label in enumerate(self.get_classes()):
            log_posterior[index_l] = np.log(self.get_priors(label))
                # retrieves the prior
            for ind_x, word in enumerate(x_attr):
                log_posterior[index_l] += x_new[ind_x] * self.get_log_densities(label)[word]
                    # the unnormalized log-posterior is progressively computed
        marginal = np.sum(np.exp(log_posterior))
            # computing the marginal density of x_new
        if marginal != 0:
            log_posterior = log_posterior - np.log(marginal)
                # array-wise computing the proper posterior probabilities
        return log_posterior

    def predict_one(self, x_new):
        if x_new.shape[0] > 1:
            raise ValueError('Too many new observations to predict!')
        post = self.calc_posteriors(x_new) # posteriors calculation
        y_pred = str(self.get_classes()[np.argmax(post)])
        p_hat = float( np.round(np.exp(np.max(post)), 5) )
        return y_pred, p_hat

    def predict_batch(self, xx_new):
        if xx_new.shape[0] == 1:
            wrn.warn("For predicting one new observation it is recommended to use predict_one()")
        y_pred = [0 for i in range(xx_new.shape[0])]
            # creating a list to accommodate the predictions
        for index, x in enumerate(xx_new):
            pred_x = self.predict_one(x)[0]
            y_pred[index] = pred_x
        return y_pred

    def predict_posteriors(self, x_new):
        P = {}
        post = self.calc_posteriors(x_new)
        y_pred = str( self.get_classes()[np.argmax(post)] )
        for index, label in enumerate(self.get_classes()):
            P[str(label)] = float( np.round(np.exp(post[index]), 5) )
        return y_pred, P

# Toy example
toy_labels = ['Dutch','English', 'Italian']

x_train_toy = [
    "Het-weer-is-mooi-vandaag!",
    "Ben ik een1 goed persoo00n?",
    "The Weather is nice Today...",
    "You Are Good",
    "Oggi fa freddo1!",
    "SeI Un Br34AvO UoMo"
]
y_train_toy = ["Dutch","Dutch","English","English","Italian","Italian"]

data_train_toy = pd.DataFrame({'Text':x_train_toy, 'Language': y_train_toy})

x_test_toy = [
    "Vandaag ben ik heel Goed goed",
    "I AM GOOD AND NICE WITH YOU!?",
    "sono un UOMO freddo",
]
y_test_toy = ["Dutch", "English", "Italian"]

data_train_toy

# Function to pre-process every text
def text_preprocess(corpus):

    remove_table = str.maketrans('', '', string.punctuation + string.digits)
        # define a translation table to remove punctuation and digits
    cleaned = []
    for doc in corpus:
        # iterate over each sentence in the list and clean it
        doc = doc.lower() # lower-case
        doc = doc.replace('-', ' ')
        doc = doc.replace('–', ' ')
        doc = doc.translate(remove_table)
        cleaned.append(doc)

    return cleaned


x_train_toy = text_preprocess(x_train_toy)
x_test_toy = text_preprocess(x_test_toy)

print(x_train_toy)
print(x_test_toy)

def my_count_vect(x_train):
    token_doc = [doc.split() for doc in x_train]
    # nested list with the tokenized words from every document, respectively

    vocabulary = set() # to store only unique values
    for doc in token_doc:
        for word in doc:
            if len(word) >= 2: # ignores the 1-letter words
                vocabulary.add(word)
    vocabulary = sorted(vocabulary)

    count_matrix = []
    for doc in token_doc:
        word_occ = [doc.count(word) for word in vocabulary]
            # counts the occurrences of the selected word in the selected document
        count_matrix.append(word_occ) # builds the count matrix
    return vocabulary, count_matrix

def my_count_vect_new(x_test, vocabulary):
    new_token_doc = [doc.split() for doc in x_test]
        # tokenize the new documents
    new_count_matrix = []
    for doc in new_token_doc:
        new_word_occ = [doc.count(word) for word in vocabulary]
            # create the count vector for each new document based on the training vocabulary
        new_count_matrix.append(new_word_occ)

    return new_count_matrix


vocabulary, x_train_toy = my_count_vect(x_train_toy)
print(vocabulary)
print(np.array(x_train_toy))
np.array(x_train_toy).shape

x_test_toy = my_count_vect_new(x_test_toy, vocabulary)
print(np.array(x_test_toy))
np.array(x_test_toy).shape


x_train_toy = csr_matrix(x_train_toy)
x_test_toy = csr_matrix(x_test_toy)


nb_toy = myMNB()
nb_toy.fit(x_train_toy, y_train_toy)

nb_toy.priors

np.exp(nb_toy.get_log_densities('English'))

y_toy_pred = nb_toy.predict_batch(x_test_toy)
print(y_toy_pred)


nb_toy.predict_posteriors(x_test_toy[0])
nb_toy.predict_posteriors(x_test_toy[1])
nb_toy.predict_posteriors(x_test_toy[2])


print('Accuracy Score:', accuracy_score(y_test_toy, y_toy_pred))
conf_m = confusion_matrix(y_test_toy, y_toy_pred, labels = toy_labels)
plt.figure(figsize=(3, 3))  # Adjust size to accommodate 8x8 matrix
sbn.heatmap(conf_m, annot=True, fmt="d",
            xticklabels=toy_labels, yticklabels=toy_labels, cmap = plt.cm.gray_r)
            # cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

############################################
# Real-life example
splits = {'train': 'WiLI-2018 dataset/train-00000-of-00001.parquet', 'test': 'WiLI-2018 dataset/test-00000-of-00001.parquet'}
df1 = pd.read_parquet("hf://datasets/MartinThoma/wili_2018/" + splits["train"])
df2 = pd.read_parquet("hf://datasets/MartinThoma/wili_2018/" + splits["test"])

# lng_labels = {
#             31: 'German',
#             39: 'Portuguese',
#             41: 'English',
#             62: 'Danish',
#             82: 'French',
#             86: 'Finnish',
#             93: 'Swedish',
#             126: 'Italian',
#             185: 'Dutch',
#             188: 'Spanish'
# }

# lng_labels = {
#     0: 'Min Dong',
#     1: 'Gilaki',
#     2: 'Jamaican Patois',
#     3: 'Luganda',
#     4: 'Sanskrit',
#     5: 'Rusyn',
#     6: 'Wolof',
#     7: 'Newari',
#     8: 'Mirandese',
#     9: 'Breton',
#     10: 'Arabic',
#     11: 'Armenian',
#     12: 'Mingrelian',
#     13: 'Extremaduran',
#     14: 'Cornish',
#     15: 'Yoruba',
#     16: 'Dhivehi',
#     17: 'Assamese',
#     18: 'Latin',
#     19: 'Welsh',
#     20: 'Fiji Hindi',
#     21: 'Achinese',
#     22: 'Kabardian',
#     23: 'Tajik',
#     24: 'Russian',
#     25: 'Northern Sotho',
#     26: 'Burmese',
#     27: 'Malay',
#     28: 'Avar',
#     29: 'Chavacano',
#     30: 'Urdu',
#     31: 'German',
#     32: 'Swahili (macrolanguage)',
#     33: 'Pushto',
#     34: 'Buryat',
#     35: 'Udmurt',
#     36: 'Kashubian',
#     37: 'Yiddish',
#     38: 'Võro',
#     39: 'Portuguese',
#     40: 'Pennsylvania German',
#     41: 'English',
#     42: 'Thai',
#     43: 'Haitian Creole',
#     44: 'Lombard',
#     45: 'Pangasinan',
#     46: 'Javanese',
#     47: 'Chuvash',
#     48: 'nan',
#     49: 'Scots',
#     50: 'Georgian',
#     51: 'Bhojpuri',
#     52: 'Bosnian',
#     53: 'Konkani',
#     54: 'Ossetian',
#     55: 'Maori',
#     56: 'Western Frisian',
#     57: 'Catalan',
#     58: 'South Azerbaijani',
#     59: 'Kinyarwanda',
#     60: 'Hindi',
#     61: 'Shona',
#     62: 'Danish',
#     63: 'Emilian',
#     64: 'Macedonian',
#     65: 'Romanian',
#     66: 'Bulgarian',
#     67: 'Croatian',
#     68: 'Somali',
#     69: 'Pampanga',
#     70: 'Navajo',
#     71: 'Ripuarisch',
#     72: 'Classical Nahuatl',
#     73: 'Central Khmer',
#     74: 'Samogitian',
#     75: 'Sranan',
#     76: 'Bavarian',
#     77: 'Corsican',
#     78: 'Central Kurdish',
#     79: 'Palatine German',
#     80: 'Egyptian Arabic',
#     81: 'Tarantino dialect',
#     82: 'French',
#     83: 'Maithili',
#     84: 'Cantonese',
#     85: 'Gujarati',
#     86: 'Finnish',
#     87: 'Kirghiz',
#     88: 'Volapük',
#     89: 'Hausa',
#     90: 'Afrikaans',
#     91: 'Uighur',
#     92: 'Lao',
#     93: 'Swedish',
#     94: 'Slovene',
#     95: 'Korean',
#     96: 'Silesian',
#     97: 'Serbian',
#     98: 'Doteli',
#     99: 'Narom',
#     100: 'Lower Sorbian',
#     101: 'Indonesian',
#     102: 'Walloon',
#     103: 'Western Panjabi',
#     104: 'Ukrainian',
#     105: 'Bishnupriya',
#     106: 'Vietnamese',
#     107: 'Turkish',
#     108: 'Aymara',
#     109: 'Lithuanian',
#     110: 'Zeeuws',
#     111: 'Polish',
#     112: 'Estonian',
#     113: 'Sicilian',
#     114: 'Vlaams',
#     115: 'Saterfriesisch',
#     116: 'Gagauz',
#     117: 'Guarani',
#     118: 'Kazakh',
#     119: 'Bengali',
#     120: 'Picard',
#     121: 'Banjar',
#     122: 'Karachay-Balkar',
#     123: 'Amharic',
#     124: 'Dimli',
#     125: 'Luxembourgish',
#     126: 'Italian',
#     127: 'Kabyle',
#     128: 'Belarusian',
#     129: 'Old English ',
#     130: 'Eastern Mari',
#     131: 'Chechen',
#     132: 'Komi-Permyak',
#     133: 'Manx',
#     134: 'Ido',
#     135: 'Faroese',
#     136: 'Bashkir',
#     137: 'Icelandic',
#     138: 'Central Bikol',
#     139: 'Tetum',
#     140: 'Japanese',
#     141: 'Kurdish',
#     142: 'Banyumasan',
#     143: 'Tuvan',
#     144: 'Livvi-Karelian',
#     145: 'Aragonese',
#     146: 'Oriya',
#     147: 'Limburgan',
#     148: 'Telugu',
#     149: 'Lingala',
#     150: 'Romansh',
#     151: 'Albanian',
#     152: 'Xhosa',
#     153: 'Malagasy',
#     154: 'Persian',
#     155: 'Serbo-Croatian',
#     156: 'Tamil',
#     157: 'Azerbaijani',
#     158: 'Ladino',
#     159: 'Bokmål',
#     160: 'Sinhala',
#     161: 'Scottish Gaelic',
#     162: 'Neapolitan',
#     163: 'Sindhi',
#     164: 'Asturian',
#     165: 'Malayalam',
#     166: 'Moksha',
#     167: 'Tswana',
#     168: 'Low German',
#     169: 'Tagalog',
#     170: 'Norwegian Nynorsk',
#     171: 'Sundanese',
#     172: 'Literary Chinese',
#     173: 'Lojban',
#     174: 'Crimean Tatar',
#     175: 'Papiamento',
#     176: 'Occitan',
#     177: 'Hakka Chinese',
#     178: 'Uzbek',
#     179: 'Standard Chinese',
#     180: 'Upper Sorbian',
#     181: 'Northern Sami',
#     182: 'Maltese',
#     183: 'Veps',
#     184: 'Lezghian',
#     185: 'Dutch',
#     186: 'West Low German',
#     187: 'Western Mari',
#     188: 'Spanish',
#     189: 'Cebuano',
#     190: 'Interlingua',
#     191: 'Hebrew',
#     192: 'Hungarian',
#     193: 'Quechua',
#     194: 'Karakalpak',
#     195: 'Marathi',
#     196: 'Venetian',
#     197: 'Arpitan',
#     198: 'Modern Greek',
#     199: 'Yakut',
#     200: 'Basque',
#     201: 'Czech',
#     202: 'Slovak',
#     203: 'Cherokee',
#     204: 'Ligurian',
#     205: 'Nepali (macrolanguage)',
#     206: 'Sardinian',
#     207: 'Iloko',
#     208: 'Belarusian (Taraschkewiza)',
#     209: 'Tibetan',
#     210: 'Oromo',
#     211: 'Waray',
#     212: 'Galician',
#     213: 'Mongolian',
#     214: 'Irish',
#     215: 'Minangkabau',
#     216: 'Igbo',
#     217: 'Interlingue',
#     218: 'Esperanto',
#     219: 'Latvian',
#     220: 'Northern Luri',
#     221: 'Alemannic German',
#     222: 'Mazanderani',
#     223: 'Aromanian',
#     224: 'Friulian',
#     225: 'Tatar',
#     226: 'Erzya',
#     227: 'Panjabi',
#     228: 'Tongan',
#     229: 'Komi',
#     230: 'Wu Chinese',
#     231: 'Tulu',
#     232: 'Turkmen',
#     233: 'Kannada',
#     234: 'Latgalian',
# }


lng_labels = {
    31: 'German',
    39: 'Portuguese',
    41: 'English',
    62: 'Danish',
    64: 'Macedonian',
    65: 'Romanian',
    67: 'Croatian',
    82: 'French',
    86: 'Finnish',
    93: 'Swedish',
    94: 'Slovene',
    107: 'Turkish',
    109: 'Lithuanian',
    111: 'Polish',
    112: 'Estonian',
    126: 'Italian',
    137: 'Icelandic',
    151: 'Albanian',
    159: 'Bokmål',
    182: 'Maltese',
    185: 'Dutch',
    188: 'Spanish',
    192: 'Hungarian',
    201: 'Czech',
    202: 'Slovak',
    219: 'Latvian'
}


lng_labels = dict(sorted(lng_labels.items(), key=lambda item: item[1]))

y_labels = list(lng_labels.values())
# storing the list of languages chosen to be extracted from the full dataset

lng_keep = list(lng_labels.keys())
# list of the languages to keep

print(y_labels)

data = pd.concat([df2, df1], ignore_index=True)
data.rename(columns = {'label':'Language', 'sentence':'Text'}, inplace=True)
data = data.query('Language in @lng_keep').reset_index(drop=True)
# extracting the selected languages

data.replace(lng_labels, inplace=True)
print(data.shape)
data.head()

sbn.countplot(data['Language'])
plt.xlabel('Documents count')
plt.title('Languages Balance')
plt.show()


x = data['Text']
y = data['Language']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True,
                                                   stratify = y, random_state = 42)

print('Training set length:', len(y_train))
print('Test set length:', len(y_test))


train_counts = y_train.value_counts().reset_index()
train_counts['Set'] = 'Training'
test_counts = y_test.value_counts().reset_index()
test_counts['Set'] = 'Test'
plot_df = pd.concat([train_counts, test_counts])
plot_df.columns = ['Language', 'Count', 'Set']
plt.figure(figsize=(10, 8))
# Swapping x and y for horizontal orientation
sbn.barplot(data=plot_df, x='Count', y='Language', hue='Set')
plt.xlabel('Documents count')
plt.ylabel('Language')
plt.title('Training and Test Set Balance per Language')
plt.legend()
plt.show()

x_train = text_preprocess(x_train)
x_test = text_preprocess(x_test)

vect=CountVectorizer(min_df=3)
vect.fit(x_train)
x_train = vect.transform(x_train)
x_test = vect.transform(x_test)

print(x_train.shape)
print(x_test.shape)
x_train

nb = myMNB()
nb.fit(x_train, y_train)

nb.get_classes()

print(nb.priors)

print(nb.get_log_densities('Finnish'))

lang_pred = nb.predict_batch(x_test)
print(lang_pred)

print('Accuracy Score:', accuracy_score(y_test, lang_pred))
conf_matr = confusion_matrix(y_test, lang_pred, labels = y_labels)
plt.figure(figsize=(6, 5))
sbn.heatmap(conf_matr, annot=True, fmt="d",
            xticklabels=y_labels, yticklabels=y_labels, cmap = plt.cm.gray_r)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


import pickle
# Save the trained custom Naive Bayes model
with open('nb_model.pkl', 'wb') as f:
    pickle.dump(nb, f)

# Save the fitted CountVectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vect, f)


####################################################
# See how the scikit-learn implementation behaves
from sklearn.naive_bayes import MultinomialNB
nb_built_in = MultinomialNB()
nb_built_in.fit(x_train, y_train)

lang_pred_2 = nb_built_in.predict(x_test) 
print(lang_pred_2)
print('Accuracy Score:', accuracy_score(y_test, lang_pred_2))
conf_matr = confusion_matrix(y_test, lang_pred_2, labels = y_labels)
plt.figure(figsize=(6, 5))
sbn.heatmap(conf_matr, annot=True, fmt="d",
            xticklabels=y_labels, yticklabels=y_labels, cmap = plt.cm.gray_r)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()