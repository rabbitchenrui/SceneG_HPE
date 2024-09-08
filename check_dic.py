import pickle as pkl

# 读取字典
with open('seg_dic.pkl', 'rb') as f:
    loaded_dict = pkl.load(f)
print(loaded_dict)