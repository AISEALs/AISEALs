import torch
from torch.utils.data import SequentialSampler, DataLoader
import logging

from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model import MultiModal, MultiModalV2


def inference():
    args = parse_args()
    logging.info("inference parameters: {}".format(args))
    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

    # 2. load model
    model = MultiModalV2(args)
    checkpoint = torch.load(args.cfs_base + '/' + args.savedmodel_path + '/' + args.ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
    model.eval()

    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            output = model(batch)
            pred_label_id = output['cate_pred_label']
            predictions.extend(pred_label_id.cpu().numpy())

    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')


if __name__ == '__main__':
    inference()
