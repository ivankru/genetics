#Dima's script
#/ayb/vol2/home/dumerenkov/current_paper/notebooks/1_HG_chipseq.ipynb
#/ayb/vol2/home/dumerenkov/current_paper/notebooks/1_HG_kousine.ipynb
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils import data
from joblib import load
import numpy as np
import os
from transformers import BertForSequenceClassification, BertConfig, PreTrainedTokenizer, BasicTokenizer, BertForTokenClassification
import collections
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
from scipy.signal import convolve
import torch
import torch.nn as nn
#from train import Descriminator
from sklearn.metrics import roc_auc_score, classification_report
#from obsolete.classificator import Classificator

import warnings
warnings.filterwarnings("ignore")

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


class Dataset(data.Dataset):
    def __init__(self, chroms, features, 
                 dna_source, features_source, 
                 labels_source, intervals, tokenizer, augmentation=None):

        self.chroms = chroms
        self.features = features
        self.dna_source = dna_source
        self.features_source = features_source
        self.labels_source = labels_source
        self.intervals = intervals
        self.augmentation = augmentation
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
        
#         ZHUNT PART
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
            if np.random.rand(1) < self.augmentation:
                ll = augmentation(ll, rate=0.1, max_shift=50)
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



def classification_metrics(data_loader, net, loss_func, device):
    net.eval()
    label_list = None
    prediction_list = None
    loss_list = []
    with torch.no_grad():
        for batch in data_loader:
            _, labels, _, input_ids, _ = batch
            #labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            input_ids = input_ids.type(torch.LongTensor)
            input_ids = input_ids.to(device)
            labels = labels.sum(dim=1)
            labels[labels > 0] = 1
            prediction = net(input_ids)["logits"]
            prediction = prediction.diff(dim=1) #["logits"]
            prediction = prediction.squeeze(1)
            labels = labels.type(torch.FloatTensor)
            labels = labels.to(device)
            loss = loss_func(prediction, labels)
            labels = labels.type(torch.LongTensor)
            loss_list.append(loss.item())
            prediction = prediction.detach().cpu().numpy()
            if label_list is not None:
                label_list = np.hstack([label_list, labels.detach().cpu().numpy()])
            else:
                label_list = labels.detach().cpu().numpy()
            if prediction_list is not None:
                prediction_list = np.hstack([prediction_list, prediction])
            else:
                prediction_list = prediction
    rocauc = roc_auc_score(label_list, prediction_list)
    prediction_list = torch.sigmoid(torch.FloatTensor(prediction_list))
    prediction_list = np.round(prediction_list.numpy())
    rep_dict = classification_report(label_list, prediction_list, output_dict=True)
    mean_loss = sum(loss_list) / len(loss_list)
    return rocauc, rep_dict, mean_loss


def augmentation(x, rate=0.1, max_shift=50, p_break=0.2):
    n = len(x)
    # if np.random.rand(1) < p_transform:
    #     x = list(reversed(x))
    if np.random.rand(1) < p_break:
        x = x[len(x) // 2:] + x[:len(x) // 2]
    #elements deletion
    shift = np.random.randint(max_shift)
    indices_to_delete = np.random.choice(np.arange(len(x)-1), shift, replace=False)
    for i in sorted(indices_to_delete, reverse=True):
        if i >= len(x) - 1:
            print(sorted(indices_to_delete, reverse=True), i)
        else:
            del x[i]
    #elements insertion
    indices_to_insert = np.random.choice(np.arange(len(x)-1), shift)    
    new_elements = np.random.choice(["A", "C", "T", "G"], shift) 
    for i, idx in enumerate(sorted(indices_to_insert, reverse=False)):
        x.insert(idx + i, new_elements[i])
    #changing elements
    x = np.array(x)
    n_to_augment = np.random.randint(int(np.round(rate * n)))
    positions = np.random.randint(low=0, high=n, size=n_to_augment)
    random_chrom = np.random.choice(["A", "C", "T", "G"], n_to_augment)
    x[positions] = random_chrom
    x = list(x)
    return x



if __name__ == "__main__":
    device = "cuda:1"
    batch_size = 32
    folder_path = "/gim/lv01/dumerenkov/zdna_data/datasets"
    tokenizer = DNATokenizer.from_pretrained('/home/kruzhilov/genetics/6-new-12w-0/')

    MODEL_NUMBER = 1
    #file_name = f'ds_w_seq_hg_fold{MODEL_NUMBER}.pkl'
    #file_name = f'ds_w_seq_hg_fold{MODEL_NUMBER}_kouzine.pkl'
    #file_name = f'ds_w_seq_mm_fold{MODEL_NUMBER}_chipseq.pkl'
    file_name = f'ds_w_seq_mm_fold{MODEL_NUMBER}_kouzine.pkl'
    file_path = os.path.join(folder_path, file_name)
    print("file path:", file_path)
    train_dataset, test_dataset = load(file_path)

    train_dataset = Dataset(train_dataset.chroms, train_dataset.features, train_dataset.dna_source, train_dataset.features_source,
    train_dataset.labels_source,  train_dataset.intervals, train_dataset.tokenizer, augmentation=0.5)
    weights = torch.FloatTensor(train_dataset.samples_weight)
    sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_dataset = Dataset(train_dataset.chroms, test_dataset.features, test_dataset.dna_source, train_dataset.features_source,
    test_dataset.labels_source,  test_dataset.intervals, test_dataset.tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)
    
    #descr = Classificator()
    config = BertConfig.from_pretrained('https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-6/config.json')
    dir_to_pretrained_model = "6-new-12w-0/"
    descr = BertForSequenceClassification.from_pretrained(dir_to_pretrained_model, config=config)

    descr.to(device)
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(descr.parameters(), lr=0.0001, weight_decay=0.05)

    for epoch in range(1):
        loss_list = []
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            features, labels, sequences, input_ids, intervals = batch
            labels = labels.type(torch.FloatTensor)
            #labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            # sequences = torch.transpose(sequences, 1, 2) 
            # sequences = sequences.type(torch.FloatTensor)
            # sequences = sequences.to(device)
            input_ids = input_ids.type(torch.LongTensor)
            input_ids = input_ids.to(device)
            labels = labels.sum(dim=1)
            labels[labels <= 10] = 0
            labels[labels > 10] = 1
            prediction = descr(input_ids)
            prediction = prediction["logits"].diff(dim=1) #["logits"]
            prediction = prediction.squeeze(1)
            loss = loss_func(prediction, labels)
            loss_list.append(loss.item())
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=descr.parameters(), max_norm=0.1)
            loss.backward()
            optimizer.step()
            #break
            if idx % 200 == 0:
                mean_loss = sum(loss_list) / len(loss_list)
                loss_list = []
                rocauc, rep_dict, mean_val_loss = classification_metrics(test_loader, descr, loss_func, device)
                formated_string = ("{epoch} train loss:{tr_loss:.4f}, val_loss:{val_loss:.4f}, " + \
                    "f1_0:{f1_0:.2f}%, f1_1:{f1_1:.2f}%, " + \
                    "rocauc:{rocauc:.2f}%").format(epoch=epoch, tr_loss=mean_loss, \
                f1_0=100*rep_dict["0"]["f1-score"], f1_1=100*rep_dict["1"]["f1-score"], rocauc=100*rocauc, val_loss=mean_val_loss)
                print(formated_string) #"val_loss:", mean_loss_val,

        if rep_dict["1"]["f1-score"] > 0.9:
            path_to_save = "/ayb/vol1/kruzhilov/weights/genomes/dna_bert.pth"
            torch.save(descr.state_dict(), path_to_save)