import os

path = "/home/lij/PycharmProjects/Seg/Final_test/image"
l = {}
key = []
ps=[]
for p in os.listdir(path):
    k = p.split('][')[0][1:]
    k = k.split('_')[0]
    if '20221009112654' in p:
        ps.append(p)
    if k not in l:
        l[k] = 1
        key.append(k)
    else:
        l[k] += 1
print(l)
print(ps)

path2 = "/mnt/algo_storage_server/UNet/Data/Checked/"
l2 = {}
key2 = []
ps2=[]
for p in os.listdir(path2):
    k = p.split('][')[0][1:]
    k = k.split('_')[0]
    if '20221009112654' in p:
        ps2.append(p)
    if k not in l2:
        l2[k] = 1
        key2.append(k)
    else:
        l2[k] += 1
print(l2)

for k in key:
    if k not in key2:
        print(k)
    else:
        if l[k] != l2[k]/2:
            print(k, l[k]*2, l2[k])
print(ps2)
for i in ps:
    if i not in ps2:
        print(i)