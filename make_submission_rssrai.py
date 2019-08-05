#!/usr/bin/env python3
import argparse
import csv
import json
import gzip
from functools import partial
from pathlib import Path
from multiprocessing.pool import Pool
import traceback
from typing import List, Tuple, Set

import cv2
import numpy as np

import utils_rssrai
from train_rssrai import Model, HyperParams, Image


logger = utils_rssrai.get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('logdir', type=Path, help='Path to log directory')
    #arg('output', type=str, help='Submission csv')
    arg('--only', help='Only predict these image ids (comma-separated)')
    arg('--threshold', type=float, default=0.5)
    arg('--epsilon', type=float, default=2.0, help='smoothing')
    arg('--min-area', type=float, default=50.0)
    arg('--min-small-area', type=float, default=10.0)
    arg('--masks-only', action='store_true', help='Do only mask prediction')
    arg('--model-path', type=Path,
        help='Path to a specific model (if the last is not desired)')
    arg('--processes', type=int, default=30)
    arg('--validation',  action='store_true',
        help='only validation images, check jaccard, '
             'save masks and polygons as png')
    arg('--valid-polygons', action='store_true', help='validation via polygons')
    arg('--force-predict', action='store_true')
    arg('--no-edges', action='store_true', help='disable prediction on edges')
    arg('--buffer', type=float, help='do .buffer(x) on pred polygons')
    args = parser.parse_args()
    hps = HyperParams(**json.loads(
        args.logdir.joinpath('hps.json').read_text()))

    only = set(args.only.split(',')) if args.only else set()

    store = args.logdir  # type: Path

    test_ids = set(['GF2_PMS1__20150902_L1A0001015646-MSS1',
            'GF2_PMS1__20150902_L1A0001015648-MSS1',
            'GF2_PMS1__20150912_L1A0001037899-MSS1',
            'GF2_PMS1__20150926_L1A0001064469-MSS1',
            'GF2_PMS1__20160327_L1A0001491484-MSS1',
            'GF2_PMS1__20160430_L1A0001553848-MSS1',
            'GF2_PMS1__20160623_L1A0001660727-MSS1',
            'GF2_PMS1__20160627_L1A0001668483-MSS1',
            'GF2_PMS1__20160704_L1A0001680853-MSS1',
            'GF2_PMS1__20160801_L1A0001734328-MSS1'])

    val_ids = set(['GF2_PMS1__20160421_L1A0001537716-MSS1',
                   'GF2_PMS2__20150217_L1A0000658637-MSS2'])

    if only:
        to_predict = only
    elif args.validation:
            to_predict = set(val_ids)
    else:
        to_predict = set(test_ids)

    if not args.force_predict:
        to_predict_masks = [
            im_id for im_id in to_predict if not mask_path(store, im_id).exists()]
    else:
        to_predict_masks = to_predict

    if to_predict_masks:
        predict_masks(args, hps, store, to_predict_masks, args.threshold,
                      validation=args.validation, no_edges=args.no_edges)
    if args.masks_only:
        logger.info('Was building masks only, done.')
        return


def mask_path(store: Path, im_id: str) -> Path:
    return store.joinpath('{}.bin-mask.gz'.format(im_id))


def predict_masks(args, hps, store, to_predict: List[str], threshold: float,
                  validation: str=None, no_edges: bool=False):
    logger.info('Predicting {} masks: {}'
                .format(len(to_predict), ', '.join(sorted(to_predict))))
    model = Model(hps=hps)
    if args.model_path:
        model.restore_snapshot(args.model_path)
    else:
        model.restore_last_snapshot(args.logdir)

    def load_im(im_id):
        data = model.preprocess_image(utils_rssrai.load_image(im_id))
        if hps.n_channels != data.shape[0]:
            data = data[:hps.n_channels]
        if validation == 'square':
            data = square(data, hps)
        return Image(id=im_id, data=data)

    def predict_mask(im):
        logger.info(im.id)
        return im, model.predict_image_mask(im.data, no_edges=no_edges)

    im_masks = map(predict_mask, utils_rssrai.imap_fixed_output_buffer(
        load_im, sorted(to_predict), threads=2))

    for im, mask in im_masks:
        assert mask.shape[1:] == im.data.shape[1:]
        with gzip.open(str(mask_path(store, im.id)), 'wb') as f:
            # TODO - maybe do (mask * 20).astype(np.uint8)
            np.save(f, mask)


    # for im, mask in utils_rssrai.imap_fixed_output_buffer(
    #         lambda _: next(im_masks), to_predict, threads=1):
    #     assert mask.shape[1:] == im.data.shape[1:]
    #     with gzip.open(str(mask_path(store, im.id)), 'wb') as f:
    #         # TODO - maybe do (mask * 20).astype(np.uint8)
    #         np.save(f, mask)


def square(x, hps):
    if len(x.shape) == 2 or x.shape[2] <= 20:
        return x[:hps.validation_square, :hps.validation_square]
    else:
        assert x.shape[0] <= 20
        return x[:, :hps.validation_square, :hps.validation_square]


if __name__ == '__main__':
    main()
