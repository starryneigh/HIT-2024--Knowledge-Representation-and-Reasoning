import jieba

seg_lists = []
with open('./data/my_text.txt', errors='ignore', encoding='utf-8') as fp:
   lines = fp.readlines()
   for line in lines:
       seg_list = jieba.cut(line)
       seg_lists.append(' '.join(seg_list))

with open('./data/my_seg_text.txt', 'w', encoding='utf-8') as ff:
    for i in range(len(seg_lists)):
        ff.write(seg_lists[i])