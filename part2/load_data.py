import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch
import sentencepiece as spm

PAD_IDX = 0

TOKENIZER_DIR = "custom_tokenizer"
VOCAB_SIZE = 1000

tokenizer = None

def initialize_tokenizer(finetune):
    global tokenizer

    if finetune:
        tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        return tokenizer

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    if os.path.exists(os.path.join(TOKENIZER_DIR, "tokenizer_config.json")):
        tokenizer = T5TokenizerFast.from_pretrained(TOKENIZER_DIR)
    else:        
        corpus_path = os.path.join(TOKENIZER_DIR, "corpus.txt")

        with open(corpus_path, "w") as f:
            for path in ["data/train.nl", "data/train.sql"]:
                with open(path, "r") as src:
                    for line in src:
                        text = line.strip()
                        if path.endswith(".sql"):
                            text = preprocess_sql(text)
                        f.write(text + "\n")
                        
        # Train SentencePiece
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=os.path.join(TOKENIZER_DIR, "spiece"),
            vocab_size=VOCAB_SIZE - 1,
            model_type="unigram",
            pad_id=0,
            eos_id=1,
            unk_id=2,
            bos_id=-1,
            hard_vocab_limit=False,
        )

        tokenizer = T5TokenizerFast(
            vocab_file=os.path.join(TOKENIZER_DIR, "spiece.model"),
            extra_ids=1,
        )
    
        tokenizer.save_pretrained(TOKENIZER_DIR)

    return tokenizer

def preprocess_sql(sql):
    return (sql
        .replace("<=", " LEQ_OP ")
        .replace(">=", " GEQ_OP ")
        .replace("<", " LT_OP ")
        .replace(">", " GT_OP ")
    )

def postprocess_sql(sql):
    return (sql
        .replace("LEQ_OP", "<=")
        .replace("GEQ_OP", ">=")
        .replace("LT_OP", "<")
        .replace("GT_OP", ">")
    )
    
class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.tokenizer = tokenizer
        self.data = self.process_data(data_folder, split)
    

    def tokenize_function(self, example, split):
        if split == "test":
            return self.tokenizer(example["input"], max_length=512, truncation=True)

        return self.tokenizer(
            example["input"],
            text_target="<extra_id_0> " + example["target"],
            max_length=512,
            truncation=True,
        )

    def process_data(self, data_folder, split):

        split = split.lower().strip()

        if split == "train":
            inputs_file = "train.nl"
            targets_file = "train.sql"
        elif split == "dev":
            inputs_file = "dev.nl"
            targets_file = "dev.sql"
        elif split == "test":
            inputs_file = "test.nl"
        else: 
            raise Exception("Train/Dev/Test only Allowed")
        
        input_lines = load_lines(os.path.join(data_folder, inputs_file))
        if split != "test":
            target_lines = load_lines(os.path.join(data_folder, targets_file))
            dataset = [{"input": inp, "target": preprocess_sql(tgt)} for inp, tgt in zip(input_lines, target_lines)]
        else:
            dataset = [{"input": inp} for inp in input_lines]

        return [self.tokenize_function(ex, split) for ex in dataset]
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence(
        [torch.as_tensor(ex["input_ids"], dtype=torch.long) for ex in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    encoder_mask = pad_sequence(
        [torch.as_tensor(ex["attention_mask"], dtype=torch.long) for ex in batch],
        batch_first=True,
        padding_value=0,
    )

    decoder_inputs = []
    decoder_targets = []
    for ex in batch:
        labels = ex["labels"]
        decoder_inputs.append(torch.as_tensor(labels[:-1], dtype=torch.long))
        decoder_targets.append(torch.as_tensor(labels[1:], dtype=torch.long))

    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    bos_id = batch[0]["labels"][0]
    initial_decoder_inputs = torch.full((len(batch), 1), bos_id, dtype=torch.long)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence(
        [torch.as_tensor(ex["input_ids"], dtype=torch.long) for ex in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    encoder_mask = pad_sequence(
        [torch.as_tensor(ex["attention_mask"], dtype=torch.long) for ex in batch],
        batch_first=True,
        padding_value=0,
    )
    
    bos_id = tokenizer.encode("<extra_id_0>", add_special_tokens=False)[0]
    initial_decoder_inputs = torch.full((len(batch), 1), bos_id, dtype=torch.long)

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # Todo
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x