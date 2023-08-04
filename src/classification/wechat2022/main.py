import logging
import os
import time
import torch

from config import parse_args
from data_helper import create_dataloaders
from model import MultiModal, MultiModalV2
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from scheduler.scheduler import build_scheduler
from optim.optim import build_optim
from network.ema import EMA


def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            output = model(batch)
            pred_label_id = output['cate_pred_label']
            cate_label = output['cate_label']
            loss = output['cate_loss']
            accuracy = output['cate_accuracy']
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(cate_label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)
    num_total_steps = min(len(train_dataloader) * args.max_epochs, args.max_steps)

    # 2. build model and optimizers
    model = MultiModalV2(args)
    optimizer = build_optim(model, args)
    scheduler = build_scheduler(optimizer, args, num_total_steps)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
    ema = EMA(model, args.ema_decay)
    ema.register()

    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            model.train()
            output = model(batch)
            loss = output['cate_loss']
            accuracy = output['cate_accuracy']
            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            ema.update()

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info("Epoch {} step {} eta {}: loss {}, accuracy {}".format(epoch, step, remaining_time, loss, accuracy))
            if step >= num_total_steps:
                break

        # 4. validation
        loss, results = validate(model, val_dataloader)
        results = {k: round(v, 4) for k, v in results.items()}
        logging.info("Epoch {} step {}: loss {}, {}".format(epoch, step, loss, results))

        # 5. save checkpoint
        mean_f1 = results['mean_f1']
        if mean_f1 > best_score:
            best_score = mean_f1
            state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
            torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                       '{}/model_epoch_{}_mean_f1_{}.bin'.format(args.cfs_base + '/' + args.savedmodel_path, epoch, mean_f1))

        # EMA
        if epoch >= args.ema_start_epoch:
            ema.apply_shadow()

            loss, results = validate(model, val_dataloader)
            results = {k: round(v, 4) for k, v in results.items()}
            logging.info("EMA-Epoch {} step {}: loss {}, {}".format(epoch, step, loss, results))
            mean_f1 = results['mean_f1']
            state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
            torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                       '{}/ema_model_epoch_{}_mean_f1_{}.bin'.format(args.cfs_base + '/' + args.savedmodel_path, epoch, mean_f1))

            ema.restore()


def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.cfs_base + '/' + args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: {}".format(args))

    train_and_validate(args)


if __name__ == '__main__':
    main()
