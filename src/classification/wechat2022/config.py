import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")

    parser.add_argument("--seed", type=int, default=2022, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')

    # ========================= Model Configs ==========================
    parser.add_argument("--initializer_range", type=float, default=0.02, help="model parameter initializer range")

    # ========================= Data Configs ==========================
    parser.add_argument('--cfs_base', type=str, default='/cfs/cfs-lq0xu8jj/wechat2022_data')
    parser.add_argument('--train_annotation', type=str, default='data/annotations/labeled.json')
    parser.add_argument('--test_annotation', type=str, default='data/annotations/test_a.json')
    parser.add_argument('--pretrain_zip_feats', type=str, default='data/zip_feats/unlabeled.zip')
    parser.add_argument('--train_zip_feats', type=str, default='data/zip_feats/labeled.zip')
    parser.add_argument('--test_zip_feats', type=str, default='data/zip_feats/test_a.zip')
    parser.add_argument('--test_output_csv', type=str, default='data/result.csv')
    parser.add_argument('--val_ratio', default=0.1, type=float, help='split 10 percentages of training data as validation')
    parser.add_argument('--batch_size', default=64, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=256, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=256, type=int, help="use for testing duration per worker")
    parser.add_argument('--prefetch', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=4, type=int, help="num_workers for dataloaders")

    # ======================== SavedModel Configs =========================
    parser.add_argument('--savedmodel_path', type=str, default='data/save/v1')
    parser.add_argument('--ckpt_file', type=str, default='model_.bin')
    parser.add_argument('--best_score', default=0.0, type=float, help='save checkpoint if mean_f1 > best_score')

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=10, help='How many epochs')
    parser.add_argument('--max_steps', default=100000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=20, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_ratio', default=0.06, type=float, help="warm up ratio parameters")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")

    # ========================== BERT =============================
    parser.add_argument('--bert_dir', type=str, default='data/chinese-roberta-wwm-ext')
    parser.add_argument('--bert_cache', type=str, default='data/cache')
    parser.add_argument('--bert_seq_length', type=int, default=10)
    parser.add_argument('--bert_learning_rate', type=float, default=3e-5)
    parser.add_argument('--bert_warmup_steps', type=int, default=5000)
    parser.add_argument('--bert_max_steps', type=int, default=30000)
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)

    # ========================== EMA =============================
    parser.add_argument('--ema_start_epoch', type=int, default=10, help="run ema epoch")
    parser.add_argument('--ema_decay', default=0.999, type=float, help="ema decay")

    # ========================== Video =============================
    parser.add_argument('--frame_embedding_size', type=int, default=768)
    parser.add_argument('--max_frames', type=int, default=32)
    parser.add_argument('--vlad_cluster_size', type=int, default=64)
    parser.add_argument('--vlad_groups', type=int, default=8)
    parser.add_argument('--vlad_hidden_size', type=int, default=1024, help='nextvlad output size using dense')
    parser.add_argument('--se_ratio', type=int, default=8, help='reduction factor in se context gating')

    # ========================== Fusion Layer =============================
    parser.add_argument('--fc_size', type=int, default=512, help="linear size before final linear")

    # ========================== Loss =============================
    parser.add_argument('--cate_loss_weight', type=float, default=1.0, help="category loss weight")
    parser.add_argument('--use_focal_loss', type=bool, default=True)

    # ========================== Learning Rate =============================
    parser.add_argument('--others_lr', type=float, default=5e-4, help="others lr")
    parser.add_argument('--bert_lr', type=float, default=5e-5, help="bert lr")

    # ========================== Network setting =============================
    parser.add_argument('--use_bert_mean_pool', type=int, default=2, help="if use mean pool for last hidden state of bert")
    parser.add_argument('--video_embedding_layer_mode', type=int, default=0,
        help="video bert embedding layer mode, 0 for different, 1 for same with text bert embedding layer")


    return parser.parse_args()
