#!/usr/bin/ python3
import numpy as np
from pathlib import Path
import gzip
import utils_rssrai
from PIL import Image
from libtiff import TIFF
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from tqdm import tqdm

logger = utils_rssrai.get_logger(__name__)

def get_mask_data(im_id):
    mask_cache = Path('log')
    mask_cache.mkdir(exist_ok=True)
    mask_data_path = mask_cache.joinpath('{}.bin-mask.gz'.format(im_id))
    if mask_data_path.exists():
        logger.info(im_id)
        with gzip.open(str(mask_data_path), 'rb') as f:
            try:
                masks = np.load(f)  # type: np.ndarray
            except Exception:
                logger.error('Error loading mask {}'.format(mask_data_path))
                raise
    masks = masks.transpose([1,2,0])

    return masks


def convert_mask_to_rgb(im_id):
    masks = get_mask_data(im_id)
    mask_h, mask_w, mask_c= masks.shape

    for i in tqdm(range(mask_h)):
        for j in range(mask_w):
            argmax_index = np.argmax(masks[i, j])

            onehot_encode = np.zeros((16))
            onehot_encode[argmax_index] = 1
            masks[i, j, :] = onehot_encode

    pred_img = utils_rssrai.onehot_to_rgb(masks, utils_rssrai.COLOR_DICT)

    return pred_img

#获取kappa系数
def get_kappa_value(im_id):
    tif = TIFF.open('/data/data_gWkkHSkq/kaggle-dstl/rssrai_dataset/train/{}_label.tif'.format(im_id))
    gt = tif.read_image()

    gt_mask = utils_rssrai.rgb_to_onehot(gt,utils_rssrai.COLOR_DICT)
    pred_mask = get_mask_data(im_id)

    pred_index = np.argmax(pred_mask,axis=-1)
    gt_index   = np.argmax(gt_mask,axis=-1)

    kappa = cohen_kappa_score(gt_index.reshape(-1,1),pred_index.reshape(-1,1), labels=[i for i in range(17)])
    logger.info("the kappa value for {}.tif is {}".format(im_id,kappa))

    return kappa


def validate_on_images(val_ids,image_type):
    kappa_sum = []
    for im_id in val_ids:
        pred_img = convert_mask_to_rgb(im_id)

        logger.info("write image {}".format(im_id))
        if image_type == ".png":
            imx = Image.fromarray(pred_img)
            imx.save("output/" + str(im_id) + "_label.png")
        else:
            tif = TIFF.open("output/" + str(im_id) + "_label.tif", mode='w')
            # to write a image to tiff file
            tif.write_image(pred_img,write_rgb=True)
            tif.close()

        kappa = get_kappa_value(im_id)
        kappa_sum.append(kappa)

    kappa_sum = np.array(kappa_sum)
    kappa_mean= kappa_sum.mean()
    logger.info("kappa mean is {}".format(kappa_mean))


def test_on_images(test_ids, image_type):
    kappa_sum = []
    for im_id in test_ids:
        pred_img = convert_mask_to_rgb(im_id)

        logger.info("write image {}".format(im_id))
        if image_type == ".png":
            imx = Image.fromarray(pred_img)
            imx.save("output/" + str(im_id) + "_label.png")
        else:
            tif = TIFF.open("output/" + str(im_id) + "_label.tif", mode='w')
            # to write a image to tiff file
            tif.write_image(pred_img, write_rgb=True)
            tif.close()

    logger.info("test finished{}")


test_ids = ['GF2_PMS1__20150902_L1A0001015646-MSS1',
            'GF2_PMS1__20150902_L1A0001015648-MSS1',
            'GF2_PMS1__20150912_L1A0001037899-MSS1',
            'GF2_PMS1__20150926_L1A0001064469-MSS1',
            'GF2_PMS1__20160327_L1A0001491484-MSS1',
            'GF2_PMS1__20160430_L1A0001553848-MSS1',
            'GF2_PMS1__20160623_L1A0001660727-MSS1',
            'GF2_PMS1__20160627_L1A0001668483-MSS1',
            'GF2_PMS1__20160704_L1A0001680853-MSS1',
            'GF2_PMS1__20160801_L1A0001734328-MSS1']

val_ids = ['GF2_PMS1__20160421_L1A0001537716-MSS1',
           'GF2_PMS2__20150217_L1A0000658637-MSS2']

image_type = ".tif"
val = False

if val:
    validate_on_images(val_ids,image_type)
else:
    test_on_images(test_ids,image_type)



