import torch
import torch.nn as nn
from transformers import DistilBertModel

class FFN(nn.Module):
    def __init__(self, dense_input=768, dense_output=1024, num_classes=21, dropout_rate=0.5):
        super(FFN, self).__init__()
        self.dense = nn.Linear(dense_input, dense_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(dense_output, num_classes)

    def forward(self, inputs):
        outputs = self.dense(inputs)
        outputs = self.relu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.classifier(outputs)
        return outputs

class HINTModel(nn.Module):
    def __init__(self, num_classes, dense_input=768, dense_output=1024, model_name='distilbert-base-uncased'):
        super(HINTModel, self).__init__()
        self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased', num_labels=21, finetuning_task="text-classification")
        self.bert_model.eval()
        self.ffn = FFN(dense_input, dense_output, num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert_model(input_ids = input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        output = self.ffn(output)
        return output