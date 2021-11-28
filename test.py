import numpy as np
import pandas as pd
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizer, BertModel, BertTokenizerFast
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import sklearn.metrics as metrics
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup



class AHS_BERT(pl.LightningModule):
    def __init__(self, drop_rate=0.2, activation=nn.ReLU(), freeze_bert=True):
        super(AHS_BERT, self).__init__()

        #Pretrained model
        self.bert = BertModel.from_pretrained('bert-base-cased')

        if freeze_bert is True:
            for param in self.bert.parameters():
                param.requires_grad = False

        #Replacement Layer/Finetuner
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(768, 1))

    def forward(self, input_ids, att_mask):
        vars = self.bert(input_ids=input_ids,
                               attention_mask=att_mask)
        target_output = vars[1]
        outputs = self.regressor(target_output)
        return outputs

    def training_step(self, batch, batch_num):
        # batch
        input_ids, attention_mask, token_type_ids, target = batch

        # fwd
        y_pred, attn = self.forward(input_ids, attention_mask, token_type_ids)

        # loss
        loss = metrics.mean_squared_error(y_true=target,y_pred=y_pred)

        # logs
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

    def configure_dataloaders(self, input_ids, att_masks, targets, batch_size):
        id_tensor = torch.tensor(input_ids)
        att_masks_tensor = torch.tensor(att_masks)
        target_tensor = torch.tensor(targets)
        dataset = TensorDataset(id_tensor, att_masks_tensor,
                                target_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True)
        return dataloader

df_all = pd.read_csv("all_data.csv")
df = df_all[['comment_text', 'toxicity']]
df = df[:50000]
x_train, x_temp, y_train, y_temp = train_test_split(df['comment_text'], df['toxicity'],
                                                                    random_state=1111,
                                                                    test_size=0.3)

x_validation, x_test, y_validation, y_test = train_test_split(x_temp, y_temp,
                                                                random_state=1111,
                                                                test_size=0.5)

# import BERT-base pretrained model
bert = BertModel.from_pretrained('bert-base-uncased')

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

#Bert only takes in a fixed length sentence, we must pad the text data.
x_train = x_train.apply(str)
seq_len = [len(i.split()) for i in x_train]
pd.Series(seq_len).hist(bins = 30)
plt.show()
#Pad

tokens_train = tokenizer(x_train.tolist(), padding='max_length', truncation=True, max_length = 150)
tokens_validation = tokenizer(x_validation.tolist(), padding='max_length', truncation=True, max_length = 150)
tokens_test = tokenizer(x_test.tolist(), padding='max_length', truncation=True, max_length = 150)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU")
else:
    raise NotImplementedError
    # device = torch.device("cpu")
    # print("CPU")

model = AHS_BERT()
#Push model to GPU
model.to(device)

training_DL = model.configure_dataloaders(tokens_train.data['input_ids'], tokens_train.data['attention_mask'],
                                           y_train.to_numpy(), 32)
validation_DL = model.configure_dataloaders(tokens_validation.data['input_ids'], tokens_validation.data['attention_mask'],
                                           y_validation.to_numpy(), 32)
#Define loss function
loss = nn.MSELoss()

#Define optimizer - Using Stochastic Gradient Descent
optimizer = optim.SGD(model.parameters(), lr=0.01)

#Being Lightning Trainer
trainer = pl.Trainer(gpus=1)
trainer.fit(model, training_DL, validation_DL)