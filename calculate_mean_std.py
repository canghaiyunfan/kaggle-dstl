from utils_rssrai import  load_image
import numpy as np


train_ids =     ['GF2_PMS1__20150212_L1A0000647768-MSS1',
                 'GF2_PMS1__20150902_L1A0001015649-MSS1',
                 'GF2_PMS1__20151203_L1A0001217916-MSS1',
                 'GF2_PMS1__20160327_L1A0001491417-MSS1',
                 'GF2_PMS1__20160816_L1A0001765570-MSS1',
                 'GF2_PMS1__20160827_L1A0001793003-MSS1',
                 'GF2_PMS2__20160225_L1A0001433318-MSS2',
                 'GF2_PMS2__20160510_L1A0001573999-MSS2'
                 ]

valid_ids = ['GF2_PMS1__20160421_L1A0001537716-MSS1',
            'GF2_PMS2__20150217_L1A0000658637-MSS2']

train_val = train_ids+valid_ids

g_mean = []
g_std = []

for im_id in train_val:
    im_data = load_image(im_id)

    mean = np.mean(im_data, axis=(0,1))
    std  = np.std(im_data, axis=(0,1))

    g_mean.append(mean)
    g_std.append(std)

g_m = np.array(g_mean).mean(axis=0)
g_s = np.array(g_std).mean(axis=0)

print(g_m)
print(g_s)