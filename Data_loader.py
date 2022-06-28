from torch.utils.data import Dataset
import torch

class Generate_torch_dataset(Dataset):
    def __init__(self, data_df, max_length, tokenizer,device,final_sentence):
        self.MAX_SEQ_LEN = max_length
        self.data = data_df
        self.tokenizer = tokenizer
        self.device = device
        self.final_sentence = final_sentence
    def __len__(self):
        return len(self.data)

    def find_sub_list(self,sl,l):
        results=[]
        sll=len(sl)
        for ind in (i for i,e in enumerate(l) if e==sl[0]):
            if l[ind:ind+sll]==sl:
                results.append((ind,ind+sll-1))
        return results

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        question = self.data.iloc[idx]['question']
        paragraph = self.data.iloc[idx]['doc_tokens']
        question_start = self.data.iloc[idx]['start_position']
        question_end = self.data.iloc[idx]['end_position']
        index_mark = idx
        answer = self.data.iloc[idx]['answer_text']
        question_ID = self.data.iloc[idx]['qas_id']
        tokenized_sentence = self.tokenizer.tokenize(paragraph, question)

        original_sentence = paragraph + question
        temp = 0
        index_mapping = []
        for index, word in enumerate(tokenized_sentence):
            if word != '[UNK]':
                index_mapping.append(word)
            else:
                word = original_sentence[temp:temp + 2]
                index_mapping.append(word)
            temp += len(word)

        self.final_sentence[str(question_ID)] = index_mapping

        token_dic = self.tokenizer(paragraph, question, max_length=self.MAX_SEQ_LEN, padding='max_length', truncation=True)
        answer_token = self.tokenizer.encode(answer)[1:-1]
        input_ids_final = token_dic['input_ids']
        attention_mask = token_dic['attention_mask']
        segment_ids = token_dic['token_type_ids']
        if question_start == 0 and question_end == 0:
            question_start, question_end = 0, 0
        elif len(self.tokenizer.tokenize(paragraph)) + len(self.tokenizer.tokenize(question)) > self.MAX_SEQ_LEN:
            question_start, question_end = 0, 0
        else:
            try:
                question_start, question_end = self.find_sub_list(answer_token, input_ids_final)[0]
            except IndexError:
                question_start, question_end = 0, 0
        input_ids_final = torch.tensor(input_ids_final).to(self.device)
        segment_ids = torch.tensor(segment_ids).to(self.device)
        attention_mask = torch.tensor(attention_mask).to(self.device)
        question_start = torch.tensor(question_start).to(self.device)
        question_end = torch.tensor(question_end).to(self.device)
        sample = {'input_ids': input_ids_final, 'token_type_ids': segment_ids, 'attention_mask': attention_mask,
                  'question_start': question_start,
                  'question_end': question_end, 'question_ID': str(question_ID)}
        return sample