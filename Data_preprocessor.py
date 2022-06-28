import Sentence_encoder
# Instantiate the sentence-level DistilBERT
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data, DataLoader, Batch
import numpy as np
import re
import torch
from sklearn import preprocessing
from transformers import AutoTokenizer, AutoModel
import random


def read_all_data(data_type):
    root_path = 'Mind_' + data_type
    be_columns = ['Impression_ID', 'User_ID', 'Time', 'History', 'Impressions']
    news_column = ['News_ID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'Title_Entities',
                   'Abstract_Entites']
    behaviors = pd.read_csv(root_path + '/behaviors.tsv', sep='\t', header=None)
    news = pd.read_csv(root_path + '/news.tsv', sep='\t', header=None)
    behaviors.columns = be_columns
    news.columns = news_column
    return behaviors, news


def merge_news_information(news_list):
    final_news_dic = {}
    for news in news_list:
        for index in range(len(news)):
            temp_info = news.iloc[0][1:5].to_dict()
            final_news_dic[news.iloc[index]['News_ID']] = temp_info
    return final_news_dic


def convert_df_to_graph(behaviors):
    behaviors = behaviors.dropna()
    final_dic = {}
    for index in tqdm(range(len(behaviors))):
        start_node = []
        end_node = []
        impression_ID = behaviors.iloc[index]['Impression_ID']
        current_user = behaviors.iloc[index]['User_ID']
        current_News = behaviors.iloc[index]['History'].split()
        impressions = behaviors.iloc[index]['Impressions']
        for index, news_id in enumerate(current_News):
            start_node.append(current_user)
            end_node.append(news_id)
        final_dic[impression_ID] = {'Impressions': impressions, 'End_node': end_node, 'Start_node': start_node}

    return final_dic


def graph_encoding(graph_dic, final_news_dic, user_embbedding_size=768, neg_ratio=0.4, final_user_dic=None, mode=None,
                   batch_size=16, encoder=None, News_dic=None):
    all_data = []
    for impression_ID in tqdm(graph_dic.keys()):
        le = preprocessing.LabelEncoder()
        start_node = graph_dic[impression_ID]['Start_node']
        end_node = graph_dic[impression_ID]['End_node']
        Node_all = start_node + end_node
        le.fit(np.unique(Node_all))
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        final_start = list(le.transform(start_node))
        final_end = list(le.transform(end_node))
        edge_index = [final_start, final_end]
        feature_all = []

        for index, Node in enumerate(list(le_name_mapping.keys())):
            regexp = re.compile(r'U[0-9]*')
            if regexp.search(Node):
                if final_user_dic is not None:
                    # If we have the User feature
                    user_embed = final_user_dic[Node]
                    user_index = index
                else:
                    # If not, we will replace user feature with all-zero matrix
                    user_embed = [0 for i in range(user_embbedding_size)]
                    user_embed[-1] = int(Node[1:])
                    user_index = index

            elif encoder is not None:
                current_news = final_news_dic[Node]
                final_content = [current_news['Title'], current_news['Abstract'], current_news['Category'],
                                 current_news['SubCategory']]
                feature_all.append(final_content)
            elif News_dic is not None:
                feature_all.append(News_dic[Node])

        if encoder is not None:
            feature_all = encoder.encode(feature_all, batch_size).tolist()

        feature_all.insert(user_index, user_embed)
        label_final = []
        impression_news_all = []
        target_label = graph_dic[impression_ID]['Impressions'].split()

        if mode == 'test':
            for news in target_label:
                current_news = final_news_dic[news]
                if encoder is not None:
                    final_content = [current_news['Title'], current_news['Abstract'], current_news['Category'],
                                     current_news['SubCategory']]
                elif News_dic is not None:
                    final_content = News_dic[news]
                impression_news_all.append(final_content)
        else:

            for news in target_label:
                current_news = final_news_dic[news[:-2]]
                if encoder is not None:
                    final_content = [current_news['Title'], current_news['Abstract'], current_news['Category'],
                                     current_news['SubCategory']]
                elif News_dic is not None:
                    final_content = News_dic[news[:-2]]
                impression_news_all.append(final_content)
                label_final.append(int(news[-1]))

            # Label truncation
            all_pos = [i for i, val in enumerate(label_final) if val == 1]
            neg_part = [i for i, val in enumerate(label_final) if val == 0]
            neg_part = neg_part[:int(len(neg_part) * neg_ratio)]
            final_index = all_pos + neg_part
            label_final = [1 for i in range(len(all_pos))] + [0 for i in range(len(neg_part))]
            temp = list(zip(final_index, label_final))
            random.shuffle(temp)
            final_index, label_final = zip(*temp)
            impression_news_all = [impression_news_all[i] for i in final_index]

        if encoder is not None:
            impression_news_all = encoder.encode(impression_news_all, batch_size)
        elif News_dic is not None:
            impression_news_all = torch.tensor(impression_news_all)

        if len(label_final) != 0:
            data = Data(x=torch.tensor(feature_all), edge_index=torch.tensor(edge_index).long(),
                        labels=torch.tensor(label_final).long(),
                        impressions_feature=impression_news_all, user_index=user_index,
                        impression_size=impression_news_all.shape[0])
        else:
            data = Data(x=torch.tensor(feature_all), edge_index=torch.tensor(edge_index).long(),
                        impressions_feature=impression_news_all, user_index=user_index,
                        impression_size=impression_news_all.shape[0])
        all_data.append(data)
    return all_data