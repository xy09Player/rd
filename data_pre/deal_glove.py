# encoding = utf-8
# author = xy
"""处理glove向量，能够用gensim导入"""


import shutil


def get_lines(file_path):
    f = open(file_path, 'r')
    c = 0
    for _ in f:
        c += 1
    return c


def prepend_line(file_in, file_out, line):
    with open(file_in, 'r') as old:
        with open(file_out, 'w') as new:
            new.write(str(line) + '\n')
            shutil.copyfileobj(old, new)


def run(file_in, file_out):
    num_lines = get_lines(file_in)
    first_line = '{} {}'.format(num_lines, 300)
    prepend_line(file_in, file_out, first_line)


if __name__ == '__main__':
    file_in = '../data/glove.840B.300d.txt'
    file_out = '../data/glove300.txt'
    run(file_in, file_out)
