import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertModel, BertTokenizerFast


class AHS_BERT(pl.LightningModule):
    def __init__(self, drop_rate=0.2, freeze=True):
        super(AHS_BERT, self).__init__()

        #Pretrained model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        #Freeze bert connected layers, drastically speed up training
        if freeze is True:
            for param in self.bert.parameters():
                param.requires_grad = False

        #Replacement Layer/Finetuner
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(768, 512),
            nn.Linear(512, 1),
            torch.nn.ReLU())

    def forward(self, input_ids, att_mask):
        vars = self.bert(input_ids=input_ids, attention_mask=att_mask)
        target_output = vars[1]
        outputs = self.regressor(target_output)
        return outputs.squeeze(-1)

    def training_step(self, batch, batch_idx):
        # batch
        input_ids, attention_mask, target = batch
        # fwd
        y_pred = self.forward(input_ids, attention_mask)
        # loss
        loss = F.mse_loss(y_pred.float(), target.float())
        return loss

    def validation_step(self, batch, batch_idx):
        # batch
        input_ids, attention_mask, target = batch
        # fwd
        y_pred = self.forward(input_ids, attention_mask)
        # loss
        loss = F.mse_loss(y_pred.float(), target.float())
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log('avg_val_loss', avg_loss)

    def test_step(self, batch, batch_idx):
        # batch
        input_ids, attention_mask, target = batch
        # fwd
        y_pred = self.forward(input_ids, attention_mask)
        # loss
        loss = F.mse_loss(y_pred.float(), target.float())
        return loss

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log('avg_test_loss', avg_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def configure_dataloaders(self, input_ids, att_masks, targets, batch_size, shuffle):
        id_tensor = torch.tensor(input_ids)
        att_masks_tensor = torch.tensor(att_masks)
        target_tensor = torch.tensor(targets)
        dataset = TensorDataset(id_tensor, att_masks_tensor,
                                target_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=shuffle)
        return dataloader

#Begin Script
df_all = pd.read_csv("all_data.csv")
df = df_all[['comment_text', 'toxicity']]
df = df[:500000]

x_train, x_temp, y_train, y_temp = train_test_split(df['comment_text'], df['toxicity'], random_state=1111, test_size=0.3)
x_validation, x_test, y_validation, y_test = train_test_split(x_temp, y_temp, random_state=1111, test_size=0.5)

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

#Bert only takes in a fixed length sentence, we must pad the text data.
x_train = x_train.apply(str)
seq_len = [len(i.split()) for i in x_train]
pd.Series(seq_len).hist(bins = 30)
plt.show()

tokens_train = tokenizer(x_train.tolist(), padding='max_length', truncation=True,
                         max_length=150, return_token_type_ids=False)
tokens_validation = tokenizer(x_validation.tolist(), padding='max_length', truncation=True,
                              max_length=150, return_token_type_ids=False)
tokens_test = tokenizer(x_test.tolist(), padding='max_length', truncation=True,
                        max_length=150, return_token_type_ids=False)

#Declare model
model = AHS_BERT()

#Load Datasets
training_DL = model.configure_dataloaders(tokens_train.data['input_ids'], tokens_train.data['attention_mask'],
                                           y_train.to_numpy().astype(float), 32, shuffle=True)
validation_DL = model.configure_dataloaders(tokens_validation.data['input_ids'], tokens_validation.data['attention_mask'],
                                           y_validation.to_numpy().astype(float), 32, shuffle=False)
test_DL = model.configure_dataloaders(tokens_test.data['input_ids'], tokens_test.data['attention_mask'],
                                           y_test.to_numpy().astype(float), 32, shuffle=False)

#Begin Lightning Trainer
trainer = pl.Trainer(gpus=1, max_epochs=5)
trainer.fit(model, training_DL, validation_DL)
trainer.test(dataloaders=test_DL)
