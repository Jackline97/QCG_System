
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class Generate_dataset(Dataset):
    def __init__(self, data,tokenizer,device, type):
        self.data = data
        self.device = device
        self.tokenizer = tokenizer
        self.type = type
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.type == 'list':
            Title = self.data[idx][0]
            Abstract = self.data[idx][1]
            cat = self.data[idx][2]
            subcat = self.data[idx][3]
        elif self.type == 'dataframe':
            Title = self.data.iloc[idx][0]
            Abstract = self.data.iloc[idx][1]
            cat = self.data.iloc[idx][2]
            subcat = self.data.iloc[idx][3]

        content_encoded_input = self.tokenizer(Title, Abstract, padding='max_length', truncation=True,
                                               max_length=128)
        cat_encoded_input = self.tokenizer(cat, subcat, padding='max_length', truncation=True,
                                           max_length=10)
        input_ids = torch.tensor(content_encoded_input['input_ids'] + cat_encoded_input['input_ids'][1:])
        attention_mask = torch.tensor(content_encoded_input['attention_mask'] + cat_encoded_input['attention_mask'][1:])
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        sample = {'input_ids': input_ids, 'attention_mask': attention_mask}
        return sample


class Sentence_encoding():
    def __init__(self, model, tokenizer,device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, final_content, batch_size, type):
        dataset = Generate_dataset(final_content, self.tokenizer, self.device, type)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        if type == 'list':
            outputs = []
            for data in dataloader:
                with torch.no_grad():
                    model_output = self.model(input_ids=data['input_ids'], attention_mask=data['attention_mask'])
                    sentence_embeddings = self.mean_pooling(model_output, data['attention_mask'])
                    outputs.append(sentence_embeddings)
            return torch.cat(outputs, dim=0)
        elif type == 'dataframe':
            outputs = {}
            temp_index = 0
            for data in tqdm(dataloader):
                with torch.no_grad():
                    model_output = self.model(input_ids=data['input_ids'], attention_mask=data['attention_mask'])
                    sentence_embeddings = self.mean_pooling(model_output, data['attention_mask'])
                    for index, i in enumerate(sentence_embeddings):
                        final_index = temp_index + index
                        outputs[final_content['News_ID'][final_index]] = sentence_embeddings[index].tolist()
                    temp_index+=batch_size
            return outputs


