# Chinese_RoBERTA's parameters are based on BERT model
from transformers import BertModel
import torch

class ROBERTA_QA_model(torch.nn.Module):
    def __init__(self, model_name, dropout_rate=0.4, hidden_size=768, num_labels=2):
        super(ROBERTA_QA_model, self).__init__()
        self.roberta = BertModel.from_pretrained(model_name)
        self.intermedia = torch.nn.Linear(hidden_size, hidden_size)
        self.a1 = torch.nn.LeakyReLU()
        self.dropour_layer1 = torch.nn.Dropout(dropout_rate)
        self.qa_outputs = torch.nn.Linear(hidden_size, num_labels)
        self.dropour_layer2 = torch.nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = self.dropour_layer1(output[0])
        logits = self.intermedia(logits)
        logits = self.a1(logits)
        logits = self.dropour_layer2(logits)
        logits = self.qa_outputs(logits)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        return start_logits, end_logits