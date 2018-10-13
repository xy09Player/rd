# encoding = utf-8
# author = xy


import pickle
from preprocess_data import organize_data
from preprocess_data import deal_data_for_train
import utils


def gen_liangci_set(file_in, file_out):
    df = organize_data(file_in)
    df = df[df['question_type'] == '数值型问题']
    df = deal_data_for_train(df)
    answers = df['answer'].values.tolist()

    # 获取 长度小于10， 存在数字，且末尾非数字的答案
    num_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    answer_tmp = []
    for a in answers:
        flag_xxx = False
        for num in num_list:
            if str(num) in a:
                flag_xxx = True
                break
        if len(a) <= 10 and flag_xxx and a[-1].isdigit() is False:
            answer_tmp.append(a)

    # 生成量词集合
    liangci_set = set()
    for a in answer_tmp:
        a_len = len(a)
        index = -1
        for i in range(-1, -a_len, -1):
            if a[i].isdigit():
                break
            else:
                index = i
        a = a[index:]
        liangci_set.add(a.strip())

    # 保留第一个量词
    liangci_set = set([utils.split_word_zh(liangci)[0].strip() for liangci in liangci_set])
    print('liangci num:%d' % len(liangci_set))

    # 输出量词集合
    with open(file_out, 'wb') as file:
        pickle.dump(liangci_set, file)


if __name__ == '__main__':
    file_in = '../data/first_question.json'
    file_out = '../data_gen/liangci_set.pkl'
    gen_liangci_set(file_in, file_out)



