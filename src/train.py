import torch.nn as nn
import torch as th
import argparse
from model.stam import STAM
from utils.utils import ConfigLoader, use_devices, save_model, get_root_dir, init_seed
from pathlib import Path
from torch.utils.data import DataLoader
import torch.optim as optim
from validate import evaluate
from data.dataset import ClipDataset
import mlflow as mf

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(args, net, train_set, val_set, test_set):
    criterion = nn.CrossEntropyLoss()
    device = th.device("cuda" if args.use_cuda else "cpu")
    net = use_devices(net, device)
    batch_size = args.batch_size
    fold_id = args.fold_id

    optimizer = optim.Adam(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    max_val_roc_auc = -1

    mf.set_experiment(experiment_name="stam_experiment")
    with mf.start_run():
        # Log all parameters to mlflow
        mf.log_params(vars(args))

        for epoch in range(1, args.max_epoch + 1):
            net.train()

            data_loader = DataLoader(
                dataset=train_set, batch_size=batch_size,
                num_workers=8, shuffle=True
            )

            for i, batch_data in enumerate(data_loader):
                optimizer.zero_grad()
                X_cpu, y_cpu, infant_id_list = batch_data

                X_gpu = X_cpu.to(device, dtype=th.float)
                y_gpu = y_cpu.to(device, dtype=th.long)

                scores = net(x=X_gpu)
                loss = criterion(scores, y_gpu)
                loss.backward()
                optimizer.step()

            train_roc_auc = evaluate(
                net=net, dataset=train_set,
                suffix_len=args.suffix_len, device="cuda"
            )

            val_roc_auc = evaluate(
                net=net, dataset=val_set,
                suffix_len=args.suffix_len, device=device
            )

            mf.log_metric("train_roc_auc", train_roc_auc, step=epoch)
            mf.log_metric("val_roc_auc", val_roc_auc, step=epoch)

            print(
                f"Epoch: {epoch:03}. [train] auc_score: {train_roc_auc:.4f}, "
                f"[val] roc_auc: {val_roc_auc:.4f}"
            )

            # save checkpoint and log test_roc_auc
            # if the result is better on the validation set
            if val_roc_auc > max_val_roc_auc:
                max_val_roc_auc = val_roc_auc
                model_file_path = str(
                    Path(
                        args.checkpoint_dir,
                        f"checkpoint_fold_{fold_id}_epoch_{epoch:03}.pth"
                    )
                )
                save_model(model_file_path=model_file_path, model=net)
                test_roc_auc = evaluate(
                    net=net, dataset=test_set,
                    suffix_len=args.suffix_len, device="cuda"
                )
                mf.log_metric("test_roc_auc", test_roc_auc)

def main():
    parser = argparse.ArgumentParser(
        description="Parse the parameters for training the model."
    )
    parser.add_argument(
        '--max-epoch', type=int, default=50,
        help='Max epoch.'
    )
    parser.add_argument('--fold-id', type=int, default=0, help='fold id')
    parser.add_argument(
        '--no-cuda', action='store_true', default=False,
        help='disables CUDA training'
    )
    parser.add_argument(
        '--clip-size', type=int, default=30, help='Size of each block.'
    )
    parser.add_argument(
        '--stride', type=int, default=20,
        help='Size of each block.'
    )
    parser.add_argument(
        '--length', type=int, default=200,
        help='Length of the skeleton to use'
    )
    parser.add_argument(
        '--channels', type=int, default=7,
        help='Number of channels to use.'
    )
    parser.add_argument('--max-hop', type=int, default=3, help='Max hop.')
    parser.add_argument('--dilation', type=int, default=1, help='Dilation.')
    parser.add_argument(
        '--z-dim', type=int, default=128,
        help='The dimensionality of the GCN output features'
    )
    parser.add_argument('--alpha-dim', type=int, default=128,
                        help='The dimensionality of alpha')
    parser.add_argument('--beta-dim', type=int, default=128,
                        help='The dimensionality of beta')
    parser.add_argument(
        '--gcn-strategy', type=str, default='uniform',
        help='Graph spatial neighbour strategy.'
    )
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay.')
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Mini-batch size for training and testing.'
    )
    parser.add_argument(
        '--dropout-context', type=float, default=0.3,
        help='Dropout probability for context.'
    )
    parser.add_argument(
        '--dropout-gcn', type=float, default=0.3,
        help='Dropout probability for graph convolutional network.'
    )
    parser.add_argument(
        '--suffix-len', type=int, default=0,
        help='Suffix length of the infant dir.'
    )

    args = parser.parse_args()
    args.use_cuda = not args.no_cuda and th.cuda.is_available()

    conf = ConfigLoader().config
    data_root = conf['data_root']
    pose_data_root = Path(data_root, f'pose_sequences')
    fold_dir = Path(data_root, f'split')
    train_label_file = Path(fold_dir, f'train_{args.fold_id:02}.txt')
    val_label_file = Path(fold_dir, f'val_{args.fold_id:02}.txt')
    test_label_file = Path(fold_dir, f'test_{args.fold_id:02}.txt')

    # Use the layout with 18 joints
    args.joints = list(range(18))

    train_set = ClipDataset(
        data_root=pose_data_root, clip_size=args.clip_size,
        stride=args.stride, length=args.length,
        label_file_path=train_label_file,
        suffix_len=args.suffix_len,
        channels=args.channels
    )
    val_set = ClipDataset(
        data_root=pose_data_root, clip_size=args.clip_size,
        stride=args.stride, length=args.length,
        label_file_path=val_label_file,
        suffix_len=args.suffix_len,
        channels=args.channels
    )

    test_set = ClipDataset(
        data_root=pose_data_root, clip_size=args.clip_size,
        stride=args.stride, length=args.length,
        label_file_path=test_label_file,
        suffix_len=args.suffix_len,
        channels=args.channels
    )

    graph_args = dict(
        layout="openpose", strategy=args.gcn_strategy,
        max_hop=args.max_hop, dilation=args.dilation
    )

    # create the model
    net = STAM(
        in_dim=args.channels, z_dim=args.z_dim,
        graph_args=graph_args,
        alpha_dim=args.alpha_dim,
        beta_dim=args.beta_dim,
        dropout_context=args.dropout_context,
        dropout_gcn=args.dropout_gcn
    )

    # init_seed to ensure the reproducibility
    init_seed(seed=0)
    
    net.apply(weights_init)

    root_dir = get_root_dir()
    checkpoint_dir = Path(root_dir, 'checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    train(args, net, train_set, val_set, test_set)
    return 0

if __name__ == '__main__':
    main()