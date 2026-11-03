import torch
import numpy as np
from lipreading.preprocess import *
from lipreading.dataset import MyDataset, pad_packed_collate


def get_preprocessing_pipelines(modality):
    # -- preprocess for the video stream
    preprocessing = {}
    # -- LRW config
    if modality == 'video':
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)
        preprocessing['train'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    RandomCrop(crop_size),
                                    HorizontalFlip(0.5),
                                    Normalize(mean, std),
                                    TimeMask(T=0.6*25, n_mask=1)
                                    ])

        preprocessing['val'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    CenterCrop(crop_size),
                                    Normalize(mean, std) ])

        preprocessing['test'] = preprocessing['val']

    elif modality == 'audio':

        preprocessing['train'] = Compose([
                                    AddNoise( noise=np.load('./data/babbleNoise_resample_16K.npy')),
                                    NormalizeUtterance()])

        preprocessing['val'] = NormalizeUtterance()

        preprocessing['test'] = NormalizeUtterance()

    return preprocessing


class ShardSampler(torch.utils.data.Sampler):
    """Evaluation sampler for DDP: rank r gets clips r, r+W, r+2W, ... Every clip is seen exactly once
    (DistributedSampler would pad with repeated clips and bias the accuracy)."""

    def __init__(self, dataset, rank, world_size):
        self.indices = list(range(rank, len(dataset), world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_data_loaders(args):
    preprocessing = get_preprocessing_pipelines( args.modality)

    # -- DDP: --batch-size is the global batch, split evenly over the processes
    world_size = getattr(args, 'world_size', 1)
    rank = getattr(args, 'rank', 0)
    batch_size = args.batch_size // world_size

    # create dataset object for each partition
    partitions = ['test'] if args.test else ['train', 'val', 'test']
    dsets = {partition: MyDataset(
                modality=args.modality,
                data_partition=partition,
                data_dir=args.data_dir,
                label_fp=args.label_path,
                annonation_direc=args.annonation_direc,
                preprocessing_func=preprocessing[partition],
                data_suffix='.npz',
                use_boundary=args.use_boundary,
                data_format=getattr(args, 'data_format', 'raw'),
                exclude_fp=getattr(args, 'exclude_list', None),
                ) for partition in partitions}
    samplers = {x: None for x in partitions}
    if world_size > 1:
        for x in partitions:
            samplers[x] = (torch.utils.data.distributed.DistributedSampler(dsets[x], num_replicas=world_size, rank=rank, shuffle=True)
                           if x == 'train' else ShardSampler(dsets[x], rank, world_size))
    dset_loaders = {x: torch.utils.data.DataLoader(
                        dsets[x],
                        batch_size=batch_size,
                        shuffle=samplers[x] is None,
                        sampler=samplers[x],
                        collate_fn=pad_packed_collate,
                        pin_memory=True,
                        num_workers=args.workers,
                        worker_init_fn=np.random.seed(1)) for x in partitions}
    return dset_loaders
