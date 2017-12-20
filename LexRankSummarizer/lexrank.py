import numpy as np
from . import tokenizer


class LexRank:
    def __init__(self, cluster, threshold=0.1, continuous=False,
                 d=0.85, tol=0.01, precalculated_idfs=None):
        self.cluster = cluster
        self.threshold = threshold
        self.continuous = continuous
        self.precalculated_idfs=precalculated_idfs
        self.d = d
        self.tol = tol

        words = np.unique(
            tokenizer.split_into_words('#'.join(cluster))
        )
        self.words = np.array(words)

        sentences = []
        sent2doc = []
        for i, document in enumerate(cluster):
            for sentence in tokenizer.split_into_sentences(document):
                sent = list(map(str.lower, tokenizer.split_into_words(sentence)))
                sent_vectorized = []
                for word in sent:
                    sent_vectorized.append(np.where(words == word)[0][0])
                sentences.append(sent_vectorized)
                sent2doc.append(i)

        self.sentences = sentences
        self.sent2doc = sent2doc

        self.docs_num = len(cluster)
        self.sents_num = len(sentences)
        self.words_num = words.shape[0]

        bag_of_words = np.zeros((self.sents_num, self.words_num))
        for i, sentence in enumerate(sentences):
            for word in sentence:
                bag_of_words[i, word] += 1
        self.bag_of_words = bag_of_words

    def calc_tf(self):
        tf = self.bag_of_words.copy()

        for i, sentence in enumerate(self.sentences):
            tf[i] /= len(sentence)

        return tf

    def calc_idf(self):
        words_idf = np.zeros((self.docs_num, self.words_num))

        for i, sentence in enumerate(self.sentences):
            for word in sentence:
                words_idf[self.sent2doc[i], word] = 1

        words_idf = np.log(self.docs_num / words_idf.sum(axis=0))

        if self.precalculated_idfs is not None:
            for i, word in enumerate(self.words):
                if word in self.precalculated_idfs.keys():
                    words_idf[i] = self.precalculated_idfs[word]

        idf = np.zeros((self.sents_num, self.words_num))

        for i, sentence in enumerate(self.sentences):
            for word in sentence:
                idf[i, word] = words_idf[word]

        return idf

    def cosine_metric(self):
        tf = self.calc_tf()
        idf = self.calc_idf()
        tf_idf = tf*idf

        norms = np.linalg.norm(tf_idf, axis=-1)

        cos_m = np.dot(tf_idf, tf_idf.T) / np.dot(norms[:, None], norms[None, :])

        return cos_m

    def get_salience(self):
        cos_m = self.cosine_metric()

        if self.threshold is not None:
            cos_m[cos_m < self.threshold] = 0

        if not self.continuous:
            cos_m[cos_m > 0] = 1

        cos_m /= np.sum(cos_m, axis=-1)[:, None]

        m = self.d/self.sents_num + (1-self.d)*cos_m

        p = np.ones(self.sents_num) / self.sents_num
        old_p = p + self.tol + 1
        while np.max(np.fabs(p - old_p)) > self.tol * 1 / self.sents_num:
            old_p = p
            p = np.dot(m.T, p)

        return p

    def summarize(self, sent_num):
        rank = self.get_salience()

        idx = []
        for i in np.argsort(rank):
            if len(self.sentences[i]) > 10:
                idx.append(i)
        idx = idx[-sent_num:]
        sents = []
        for i, sentence in enumerate(self.sentences):
            if i in idx:
                sents.append(' '.join(self.words[sentence]))

        return '. '.join(sents) + '.'
