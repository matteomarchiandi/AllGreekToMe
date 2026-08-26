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