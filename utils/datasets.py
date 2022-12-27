import os
import numpy as np
from scipy.signal import convolve
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import collections
import torch
from torch.utils import data
from transformers import PreTrainedTokenizer, BasicTokenizer


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {"vocab_file": {"dna3": "https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-3/vocab.txt",
                                             "dna4": "https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-4/vocab.txt",
                                             "dna5": "https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-5/vocab.txt",
                                             "dna6": "https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-6/vocab.txt"}}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
                                          "dna3": 512,
                                          "dna4": 512,
                                          "dna5": 512,
                                          "dna6": 512}

PRETRAINED_INIT_CONFIGURATION = {
    "dna3": {"do_lower_case": False},
    "dna4": {"do_lower_case": False},
    "dna5": {"do_lower_case": False},
    "dna6": {"do_lower_case": False}}


VOCAB_KMER = {
    "69": "3",
    "261": "4",
    "1029": "5",
    "4101": "6",}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def seq2kmer(seq, k):
    """
    Convert original sequence to kmers
    
    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.
    
    Returns:
    kmers -- str, kmers separated by space
    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers


def onehot_to_str(onehot):
    genes = np.array(["A", "C", "T", "G"])
    numbers = np.argmax(onehot, axis=1) 
    dna = "".join(list(genes[numbers]))
    return dna


class Dataset(data.Dataset):
    def __init__(self, chroms, features, 
                 dna_source, features_source, 
                 labels_source, intervals, tokenizer,
                 augmentation=None, p_augmentation=0.5):

        self.chroms = chroms
        self.features = features
        self.dna_source = dna_source
        self.features_source = features_source
        self.labels_source = labels_source
        self.intervals = intervals
        self.augmentation = augmentation
        self.p_augmentation = p_augmentation
        self.le = LabelBinarizer().fit(np.array([["A"], ["C"], ["T"], ["G"]]))
        self.configs = {
                        'ZHUNT_AS': {
                                'CG': 0, 'GC': 1, 'CA': 0, 'AC': 1, 
                                'TG': 0, 'GT': 1, 'TA': 1, 'AT': 1, 
                                'CC': 0, 'GG': 0, 'CT': 1, 'TC': 1, 
                                'GA': 1, 'AG': 1, 'AA': 1, 'TT': 1},
                       }
        seqs = (["A", "C", "T", "G"] + 
                ['AC', 'AT', 'AG', 'CT', 'CG', 'GT'] +
                ['AAC', 'ACC', 'AAT', 'ATT', 'AAG', 'AGG', 
                 'CCA', 'CAA', 'CCT', 'CTT', 'CCG', 'CGG', 
                 'TTA', 'TAA', 'TTC', 'TCC', 'TTG', 'TGG', 
                 'GGA', 'GAA', 'GGC', 'GCC', 'GGT', 'GTT'] +
                ['AAAC', 'AAAT', 'AAAG', 'CCCA', 'CCCT', 'CCCG',
                 'TTTA', 'TTTC', 'TTTG', 'GGGA', 'GGGC', 'GGGT'])
        self.tars = np.array([self.le.transform(list(i * 11)[:11]) for i in seqs])[:, ::-1, ::-1]
        # purine-pyrimidine
        self.tars = np.concatenate((self.tars, np.array([self.tars[4] + self.tars[9]])))
        self.tokenizer = tokenizer

        self.binary_labels = []
        for interval in self.intervals:
            label = self.labels_source[interval[0]][interval[1]:interval[2]]
            label = label.sum()
            if label > 10:
                label = 1
            else:
                label = 0
            self.binary_labels.append(label)
        self.samples_weight = np.zeros(len(self.binary_labels))
        self.binary_labels = np.array(self.binary_labels)

        n_positive = sum(self.binary_labels)
        n_negative = len(self.binary_labels) - n_positive
        self.samples_weight[self.binary_labels == 1] = n_negative / (n_negative + n_positive)
        self.samples_weight[self.binary_labels == 0] = n_positive / (n_negative + n_positive)
        self.samples_weight = self.samples_weight / sum(self.samples_weight)
        
    def __len__(self):
        return len(self.intervals)
    
    def __getitem__(self, index):
        interval = self.intervals[index]
        chrom = interval[0]
        begin = int(interval[1])
        end = int(interval[2])
        ll = list(self.dna_source[chrom][begin:end].upper())
        y = self.labels_source[interval[0]][interval[1]: interval[2]]           
        #y = self.binary_labels[index]
        
#       DNA PART     
        dna_OHE = self.le.transform(ll)[None]       
        res = pd.DataFrame(convolve(dna_OHE, self.tars)[:, 5:-5, 3].T / 11)
        res = (res.rolling(5, min_periods=1).max().values == 1).astype(int)
        
#       ZHUNT PART
        zhunts = []
        for key in self.configs:
            vec = np.array(ll)
            vec = np.vectorize(lambda x:self.configs[key].get(x, 0))(
                                    np.char.add(vec[1:], vec[:-1]))
            zhunts.append(np.concatenate([vec, [0]]))
                
        # FEATURES PART
        feature_matr = []
        for feature in self.features:
            source = self.features_source[feature]
            feature_matr.append(source[chrom][begin:end])
        
        # UNION
        if len(feature_matr) > 0:
            X = np.hstack((
                           res,
                           np.array(zhunts).T, 
                           np.array(feature_matr).T/1000)).astype(np.float32)
#             X = (np.array(feature_matr).T/1000).astype(np.float32)
        else:
            X = dna_OHE.astype(np.float32)
        
        #K-mer part
        k_mers = seq2kmer(self.dna_source[chrom][begin:end+5].upper(),6)
        encoded_k_mers = self.tokenizer.encode_plus(k_mers, add_special_tokens=False, max_length=512)["input_ids"]

        if self.augmentation:
            if np.random.rand(1) < self.p_augmentation:
                ll = self.augmentation(ll)

        return torch.Tensor(X), torch.Tensor(y).long(), self.le.transform(ll), torch.LongTensor(encoded_k_mers), (chrom, begin, end)


class DNATokenizer(PreTrainedTokenizer):
    r"""
    Constructs a BertTokenizer.
    :class:`~transformers.BertTokenizer` runs end-to-end tokenization: punctuation splitting + wordpiece
    Args:
        vocab_file: Path to a one-wordpiece-per-line vocabulary file
        do_lower_case: Whether to lower case the input. Only has an effect when do_basic_tokenize=True
        do_basic_tokenize: Whether to do basic tokenization before wordpiece.
        max_len: An artificial maximum length to truncate tokenized sequences to; Effective maximum length is always the
            minimum of this value (if specified) and the underlying BERT model's sequence length.
        never_split: List of tokens which will never be split during tokenization. Only has an effect when
            do_basic_tokenize=True
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES


    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        max_len = 512,
        **kwargs
    ):
        """Constructs a BertTokenizer.
        Args:
            **vocab_file**: Path to a one-wordpiece-per-line vocabulary file
            **do_lower_case**: (`optional`) boolean (default True)
                Whether to lower case the input
                Only has an effect when do_basic_tokenize=True
            **do_basic_tokenize**: (`optional`) boolean (default True)
                Whether to do basic tokenization before wordpiece.
            **never_split**: (`optional`) list of string
                List of tokens which will never be split during tokenization.
                Only has an effect when do_basic_tokenize=True
            **tokenize_chinese_chars**: (`optional`) boolean (default True)
                Whether to tokenize Chinese characters.
                This should likely be deactivated for Japanese:
                see: https://github.com/huggingface/pytorch-pretrained-BERT/issues/328
        """
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs
        )
        self.vocab = load_vocab(vocab_file)
        self.max_len = max_len
        #self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
        #self.max_len_sentences_pair = self.max_len - 3  # take into account special tokens

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )
        
        self.kmer = VOCAB_KMER[str(len(self.vocab))]
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case, never_split=never_split, tokenize_chinese_chars=tokenize_chinese_chars
            )

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                split_tokens.append(token)
        # print(split_tokens)
        return split_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:
            single sequence: [CLS] X [SEP]
            pair of sequences: [CLS] A [SEP] B [SEP]
        """
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]

        if token_ids_1 is None:
            if len(token_ids_0) < 510:
                return cls + token_ids_0 + sep
            else:
                output = []
                num_pieces = int(len(token_ids_0)//510) + 1
                for i in range(num_pieces):
                    output.extend(cls + token_ids_0[510*i:min(len(token_ids_0), 510*(i+1))] + sep)
                return output

        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.
        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model
        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        
        if len(token_ids_0) < 510:
            return [1] + ([0] * len(token_ids_0)) + [1]
        else:
            output = []
            num_pieces = int(len(token_ids_0)//510) + 1
            for i in range(num_pieces):
                output.extend([1] + ([0] * (min(len(token_ids_0), 510*(i+1))-510*i)) + [1])
            return output
            return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:
        0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence
        if token_ids_1 is None, only returns the first portion of the mask (0's).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            if len(token_ids_0) < 510:
                return len(cls + token_ids_0 + sep) * [0]
            else:
                num_pieces = int(len(token_ids_0)//510) + 1
                return (len(cls + token_ids_0 + sep) + 2*(num_pieces-1)) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, vocab_path):
        """Save the tokenizer vocabulary to a directory or file."""
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)

