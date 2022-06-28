from tqdm import tqdm
import torch

def replace(str1, list1):
    for token in list1:
        str1 = str1.replace(token,'')
    str1 = str1.split()
    if len(str1) ==0:
        str_final = ''
    else:
        str_final = ' '.join(str1)
    return str_final


def evaluate(model, dataloader, tokenizer,final_sentence):
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    cls_token = tokenizer.cls_token
    all_token = [sep_token,pad_token,cls_token]
    true_word = []
    pred_word = []
    current_sentence = []
    question_ID_all = []
    model.eval()

    with torch.no_grad():
        for data in tqdm(dataloader):

            source = data['input_ids']
            type_to = data['token_type_ids']
            question_start = data['question_start']
            question_end = data['question_end']
            mask = data['attention_mask']
            question_ID = data['question_ID']
            actual_sentence = [final_sentence[i] for i in question_ID]
            start_logits, end_logits = model(input_ids=source,
                                             attention_mask=mask, token_type_ids=type_to)

            answer_start = torch.argmax(start_logits, dim=1).tolist()
            answer_end = torch.argmax(end_logits, dim=1).tolist()

            final_answer_pair = list(zip(answer_start, answer_end))
            for index, sentence in enumerate(source):
                start, end = final_answer_pair[index]
                question_ID_all.append(question_ID[index])

                if int(end) == int(start) and int(end) == 0:
                    pred_word.append('')
                else:
                    pre_ans = ''.join(actual_sentence[index][int(start)-1:int(end)])
                    pred_word.append(replace(pre_ans, all_token))

                if int(question_start[index]) == int(question_end[index]) and int(question_end[index]) == 0:
                    true_word.append('')
                else:
                    tru_ans = ''.join(actual_sentence[index][int(question_start[index])-1:int(question_end[index])])
                    true_word.append(replace(tru_ans, all_token))
                sentence = ''.join(actual_sentence[index])
                current_sentence.append(replace(sentence, all_token))

    return true_word, pred_word, current_sentence, question_ID_all