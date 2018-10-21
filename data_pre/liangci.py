# encoding = utf-8
# author = xy


import pickle
from preprocess_data import organize_data


def gen_liangci_set(file_in, file_out):
    df = organize_data(file_in)
    df = df[df['question_type'] == '数值型问题']
    answers = df['article_answer'].values.tolist()

    drop_list_zh = ['。', '，', '、', '；', '：', '？', '！']
    drop_list_en = ['.', '?', '!', ';', ':', ',', '-', '...', '..', '....']
    drop_list = drop_list_zh + drop_list_en
    answers = [answer[:-1].strip() if answer[-1] in drop_list else answer for answer in answers]
    answers = [answer[1:].strip() if answer[0] in drop_list else answer for answer in answers]

    # 生成量词集合
    liangci_set = set()
    for answer in answers:
        if answer != answer or answer is None or len(answer) == 0 or answer[-1].isdigit():
            continue
        else:
            index = -1
            while index >= -len(answer) and answer[index].isdigit() is False:
                index = index - 1
            if index >= -len(answer):
                liangci = answer[index+1:]
                liangci = liangci.strip()
                if liangci != '' and len(liangci) <= 10:
                    liangci_set.add(liangci)
    print('liangci_set size:%d' % len(liangci_set))

    # 输出量词集合
    with open(file_out, 'wb') as file:
        pickle.dump(liangci_set, file)


if __name__ == '__main__':
    file_in = '../data_gen/merge.json'
    file_out = '../data_gen/liangci_set.pkl'
    gen_liangci_set(file_in, file_out)



