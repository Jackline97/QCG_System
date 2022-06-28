import json
from tqdm import tqdm
import ast

def load_Yelp_data(path):
    final_user_set = {}
    final_item_set = {}
    edge_index = {}
    edge_feature = {}

    with open(path + 'yelp_academic_dataset_user.json', 'r', encoding='utf-8') as f:
        all_data = f.readlines()
        for line in tqdm(all_data):
            current_data = json.loads(line)
            current_node = current_data['user_id']
            del current_data['user_id']
            final_user_set[current_node] = current_data

    with open(path + 'yelp_academic_dataset_business.json', 'r', encoding='utf-8') as f:
        all_data = f.readlines()
        for line in tqdm(all_data):
            current_data = json.loads(line)
            current_node = current_data['business_id']
            del current_data['business_id']
            final_item_set[current_node] = current_data

    with open(path + 'yelp_academic_dataset_review.json', 'r', encoding='utf-8') as f:
        all_data = f.readlines()
        for line in tqdm(all_data):
            current_data = json.loads(line)
            current_business = current_data['business_id']
            current_user = current_data['user_id']
            cool_level = int(current_data['funny']) + int(current_data['useful']) + int(current_data['cool'])
            current_edge_feature = {'date': current_data['date'], 'text': current_data['text'],
                                    'rate': current_data['stars'], 'rate_level': cool_level}
            edge_feature[str(current_user) + '|' + str(current_business)] = current_edge_feature
            if current_user in edge_index:
                edge_index[current_user].append(current_business)
            else:
                edge_index[current_user] = [current_business]

    return edge_index, final_user_set, final_item_set, edge_feature


def attribute_analysis(attr):
    final_str = ''
    for key in attr.keys():
        if attr[key][0] != '{' and attr[key][-1] != '}':
            final_str += '{} is {}. '.format(key, attr[key])
        else:
            values = json.loads(json.dumps(ast.literal_eval(attr[key])))
            final_str += 'The business contains {} where '.format(key)
            for sub_key in values:
                final_str+='{} is {}. '.format(sub_key, values[sub_key])
    return final_str

def feature_transformation(final_feature):
    final_feature_dic = {}
    for key_name in tqdm(list(final_feature.keys())):
        feature = final_feature[key_name]
        if 'fans' in feature.keys(): # User feature
            if len(feature['elite'])!=0:
                elite_period = max([int(i) for i in feature['elite'].split(',')]) - min([int(i) for i in feature['elite'].split(',')])
            else:
                elite_period = 0
            friends_count = len(feature['friends'].split(', '))
            valid_user_feature = [feature[i] for i in list(feature.keys())[-13:]]
            temp_feature = [feature['review_count'], feature['useful'], feature['funny'], feature['cool'],elite_period, friends_count]
            final_feature_val = temp_feature + valid_user_feature
            final_feature_dic[key_name] = final_feature_val
        else:
            name = feature['name']
            address = feature['address']
            city = feature['city']
            state = feature['state']
            postal_code = feature['postal_code']
            stars = feature['stars']
            review_count = feature['review_count']
            attribures = feature['attributes']
            categories = 'The business offers {}. '.format(feature['categories'])
            current_address_information = 'The business {} located at {},{},{},{}. '.format(name, address,city,state, postal_code)
            if attribures is not None:
                attributes_information = attribute_analysis(attribures)
            else:
                attributes_information = ''
            final_information = categories + current_address_information + attributes_information
            final_feature_dic[key_name] = final_information
    return final_feature_dic

def ten_core_user_filter(edge_idex,final_user_feature,final_item_feature, edge_feature,core_threshould, user_limit):
    new_edge_index = [[], []]
    new_edge_feature = {}
    new_user_feature = {}
    new_item_feature = {}
    user_count = 0
    for key in tqdm(edge_idex.keys()):
        current_neighbor = edge_idex[key]
        if len(current_neighbor) >=core_threshould and len(current_neighbor) <=100 and user_count<=user_limit:
            new_edge_index[0].extend([key for i in range(len(current_neighbor))])
            new_user_feature[key] = final_user_feature[key]
            user_count+=1
            for neighbor in current_neighbor:
                new_edge_index[1].append(neighbor)
                new_edge_feature[str(key) + '|' + str(neighbor)] = edge_feature[str(key) + '|' + str(neighbor)]
                new_item_feature[neighbor] = final_item_feature[neighbor]
        else:
            continue
    return new_edge_feature,new_user_feature, new_item_feature, new_edge_index