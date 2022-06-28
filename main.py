import torch
import Data_preprocessor
import os
import json
import Sentence_encoder
from transformers import AutoTokenizer, AutoModel
from torch_geometric.loader import DataLoader

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device('cpu')

print("Current experiment device is:", device)
batch_size = 128
model_name = "sentence-transformers/distilbert-base-nli-mean-tokens"

train_news_path = 'Data/news.json'
val_news_path = 'Data/val_news.json'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
encoder = Sentence_encoder.Sentence_encoding(model, tokenizer,device)

def data_preprocessor(data_type, news_embedding_path, size=None, train_mode=False, other_news_embedding=None):
    behaviors, news = Data_preprocessor.read_all_data(data_type)
    if os.path.isfile(news_embedding_path):
        with open(news_embedding_path, 'r',encoding='utf-8') as f:
            fianal_news_map = json.load(f)
    else:
        fianal_news_map = encoder.encode(news,batch_size,'dataframe')
        with open(news_embedding_path, 'w', encoding='utf-8') as f:
            json.dump(fianal_news_map, f)

    if train_mode == True and other_news_embedding is not None:
        for path in other_news_embedding:
            with open(path, 'r', encoding='utf-8') as f:
                fianal_news_map_temp = json.load(f)
                fianal_news_map.update(fianal_news_map_temp)
    # Get news' title, abstract, cat, subcat into a dic
    final_news_dic = Data_preprocessor.merge_news_information([news]) # Merge multi-source information
    # Convert dataframe to dic with (Start node, end node combination)
    if size is not None:
        final_dic = Data_preprocessor.convert_df_to_graph(behaviors[:size])
    else:
        final_dic = Data_preprocessor.convert_df_to_graph(behaviors)
    # Convert dic to torch Data graph format.
    all_data = Data_preprocessor.graph_encoding(graph_dic = final_dic,final_news_dic=final_news_dic, News_dic=fianal_news_map)
    return all_data

batch_size = 128
Train_data = data_preprocessor('train', train_news_path, size=5000)
test_data = data_preprocessor('dev', val_news_path, size=5000,train_mode=True, other_news_embedding=[train_news_path])

train_dataloader = DataLoader(Train_data,batch_size=batch_size,shuffle=False)
test_dataloader = DataLoader(test_data,batch_size=batch_size,shuffle=False)

print(next(iter(train_dataloader)))
print(next(iter(test_dataloader)))
print(next(iter(train_dataloader)).user_index)
print(next(iter(train_dataloader)).ptr)

# dynamic weight assign problme
# Cold start problem
# Scalability
# Deviation
# Drift.


