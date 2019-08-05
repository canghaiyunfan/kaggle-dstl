import os

path = '/data/data_gWkkHSkq/kaggle-dstl/rssrai_dataset/val'


for parent, dirnames, filenames in os.walk(path):
    for filename in filenames:
        os.rename(os.path.join(parent, filename), os.path.join(parent, filename.replace(' (2)', '')))
