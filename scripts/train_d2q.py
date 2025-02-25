'''
Train doc2query on MS MARCO with t5 from hgf

The training data should contains source and target in each line, which should be separate by '\t'
'''
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments
import pandas as pd
from torch.utils.data import Dataset
import argparse


class TrainerDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path, sep="\t")
        df = df.dropna()
        df.loc[1] = [str(i) for i in df.loc[1]]
        self.dataset = df
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        source = self.dataset.iloc[idx, 0]
        target = self.dataset.iloc[idx, 1]
        input_ids = self.tokenizer.encode(args.tag + ': ' + source, return_tensors='pt',
                                          padding='max_length', truncation='longest_first', max_length=512)[0]
        label = self.tokenizer.encode(str(target), return_tensors='pt', padding='max_length',
                                      truncation='longest_first', max_length=64)[0]
        return {'input_ids': input_ids, 'labels': label}


parser = argparse.ArgumentParser(description='Train docTquery on more datasets')
parser.add_argument('--pretrained_model_path', default='castorini/monot5-base-msmarco', help='pretrained model path')
parser.add_argument('--tag', default='msmarco', help='tag for training data', type=str)
parser.add_argument('--train_data_path', default='train_ar_tf_q5q6q17_v1_pairs.tsv', help='training data path')
parser.add_argument('--output_path', default='./output', help='output directory path')
parser.add_argument('--epoch', default=1, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--weight_decay', default=5e-5, type=float)
parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--save_steps', default=5, type=int)
parser.add_argument('--gra_acc_steps', default=8, type=int)
args = parser.parse_args()

model = T5ForConditionalGeneration.from_pretrained(args.pretrained_model_path)
train_dataset = TrainerDataset(args.train_data_path)


# look into Huggingface TrainingArgs and use those
# If you add a new arg to input, also set it up in the trainingArgs
training_args = TrainingArguments(
    output_dir=args.output_path,
    num_train_epochs=args.epoch,
    per_device_train_batch_size=args.batch_size,
    weight_decay=args.weight_decay,
    learning_rate=args.lr,
    gradient_accumulation_steps=args.gra_acc_steps,
    logging_dir='./logs',
    save_steps=args.save_steps
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)


train_result = trainer.train()
trainer.save_model()  # Saves the tokenizer too for easy upload

