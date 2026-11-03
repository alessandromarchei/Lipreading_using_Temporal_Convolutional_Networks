import io
import os
import atexit
import glob
import json
import torch
import random
import librosa
import numpy as np
import sys
from lipreading.utils import read_txt_lines

LMDB_FORMAT = b'lrw-npz-v1'
# Readahead on: a record is one contiguous ~170 KB npz, so kernel readahead serves it in a few reads instead of one
# 4 KB page fault at a time. Measured with tools/benchmark_dataloader.py: 2x faster on a cold cache, same when warm.
LMDB_READAHEAD = True

# One read-only handle per LMDB file and per process. py-lmdb refuses to open the same file twice in one
# process, and LMDB handles must not be used across fork(), so a forked DataLoader worker drops the handles
# it inherited and opens its own on first use.
_LMDB_ENVS = {}
_LMDB_PID = os.getpid()


def _get_lmdb_env(path):
    global _LMDB_PID
    if _LMDB_PID != os.getpid():  # forked worker (or spawn/forkserver: the dict is empty anyway)
        for env in _LMDB_ENVS.values():
            env.close()  # safe: readonly, lock=False, so there is no shared reader table to release
        _LMDB_ENVS.clear()
        _LMDB_PID = os.getpid()
    env = _LMDB_ENVS.get(path)
    if env is None:
        import lmdb  # only needed for the lmdb backend
        # lock=False: nothing writes while training, and it works on read-only mounts such as /kaggle/input.
        env = lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=LMDB_READAHEAD, meminit=False,
                        max_readers=2048)
        _LMDB_ENVS[path] = env
    return env


def _close_lmdb_envs():
    if _LMDB_PID == os.getpid():
        for env in _LMDB_ENVS.values():
            env.close()
    _LMDB_ENVS.clear()


atexit.register(_close_lmdb_envs)


class MyDataset(object):
    def __init__(self, modality, data_partition, data_dir, label_fp, annonation_direc=None,
        preprocessing_func=None, data_suffix='.npz', use_boundary=False, data_format='raw',
        exclude_fp=None):
        assert data_format in ('raw', 'lmdb'), f"Unknown data format: {data_format}"
        self._data_format = data_format
        self._exclude_fp = exclude_fp
        assert os.path.isfile( label_fp ), \
            f"File path provided for the labels does not exist. Path iput: {label_fp}."
        self._data_partition = data_partition
        self._data_dir = data_dir
        self._data_suffix = data_suffix

        self._label_fp = label_fp
        self._annonation_direc = annonation_direc

        self.fps = 25 if modality == "video" else 16000
        self.is_var_length = True
        self.use_boundary = use_boundary
        self.label_idx = -3

        self.preprocessing_func = preprocessing_func

        # -- lmdb backend: the word durations are stored inside the LMDB index, no annotation folder needed
        self._lmdb_path = None
        self._durations = {}

        if data_format == 'raw' and (self.use_boundary or (self.is_var_length and data_partition == "train")):
            assert self._annonation_direc is not None, \
                "Directory path provided for the sequence timestamp (--annonation-direc) should not be empty."
            assert os.path.isdir(self._annonation_direc), \
                f"Directory path provided for the sequence timestamp (--annonation-direc) does not exist. Directory input: {self._annonation_direc}"

        self._data_files = []

        self.load_dataset()

    def load_dataset(self):

        # -- read the labels file
        self._labels = read_txt_lines(self._label_fp)


        # -- add examples to self._data_files
        if self._data_format == 'lmdb':
            self._get_files_from_lmdb()
        else:
            self._get_files_for_partition()

        # -- drop clips listed in the exclude file (one WORD/partition/CLIP per line, e.g. from tools/scan_lmdb.py)
        if self._exclude_fp:
            excluded = {line.strip() for line in read_txt_lines(self._exclude_fp) if line.strip()}
            n_before = len(self._data_files)
            self._data_files = [f for f in self._data_files
                                if os.path.splitext('/'.join(f.split('/')[-3:]))[0] not in excluded]
            print(f"Partition {self._data_partition}: excluded {n_before - len(self._data_files)} clips listed in {self._exclude_fp}")


        # -- from self._data_files to self.list
        self.list = dict()
        self.instance_ids = dict()
        for i, x in enumerate(self._data_files):
            label = self._get_label_from_path( x )
            self.list[i] = [ x, self._labels.index( label ) ]
            self.instance_ids[i] = self._get_instance_id_from_path( x )

        print(f"Partition {self._data_partition} loaded")

    def _get_instance_id_from_path(self, x):
        # for now this works for npz/npys, might break for image folders
        instance_id = x.split('/')[-1]
        return os.path.splitext( instance_id )[0]

    def _get_label_from_path(self, x):
        return x.split('/')[self.label_idx]

    def _get_files_for_partition(self):
        # get rgb/mfcc file paths

        dir_fp = self._data_dir
        if not dir_fp:
            return

        # get npy/npz/mp4 files
        search_str_npz = os.path.join(dir_fp, '*', self._data_partition, '*.npz')
        search_str_npy = os.path.join(dir_fp, '*', self._data_partition, '*.npy')
        search_str_mp4 = os.path.join(dir_fp, '*', self._data_partition, '*.mp4')
        self._data_files.extend( glob.glob( search_str_npz ) )
        self._data_files.extend( glob.glob( search_str_npy ) )
        self._data_files.extend( glob.glob( search_str_mp4 ) )

        # If we are not using the full set of labels, remove examples for labels not used
        self._data_files = [ f for f in self._data_files if f.split('/')[self.label_idx] in self._labels ]

    def _get_files_from_lmdb(self):
        """Read the sample index of <data_dir>/<partition>.lmdb (written by preprocessing/lrw_lmdb.py).

        Each sample becomes the pseudo path WORD/<partition>/STEM.npz, so the label, instance id and
        filtering logic of the raw backend apply unchanged.
        """
        self._lmdb_path = os.path.join(self._data_dir, f'{self._data_partition}.lmdb')
        assert os.path.isfile(self._lmdb_path), f"LMDB file does not exist. Path input: {self._lmdb_path}"
        with _get_lmdb_env(self._lmdb_path).begin() as txn:
            fmt = txn.get(b'__format__')
            index = json.loads(txn.get(b'__index__'))
            lmdb_labels = json.loads(txn.get(b'__labels__'))
        assert fmt == LMDB_FORMAT, f"Unsupported LMDB format {fmt!r} in {self._lmdb_path}, expected {LMDB_FORMAT!r}"

        for key, label_id, duration in index:
            word, stem = key.split('/')
            # class ids are always recomputed from --label-path (below); check the stored ones agree
            assert lmdb_labels[label_id] == word, f"Inconsistent label id for {key} in {self._lmdb_path}"
            pseudo_path = f'{word}/{self._data_partition}/{stem}.npz'
            self._data_files.append(pseudo_path)
            self._durations[pseudo_path] = float(duration)

        # If we are not using the full set of labels, remove examples for labels not used
        self._data_files = [ f for f in self._data_files if f.split('/')[self.label_idx] in self._labels ]

    def _load_data_lmdb(self, filename):
        word, _, stem = filename.split('/')
        key = f'v:{word}/{os.path.splitext(stem)[0]}'.encode('utf-8')
        with _get_lmdb_env(self._lmdb_path).begin() as txn:
            payload = txn.get(key)
        if payload is None:
            raise KeyError(f"Key {key!r} missing from {self._lmdb_path}")
        with np.load(io.BytesIO(payload)) as npz:
            return npz['data']

    def _get_duration(self, filename):
        """Word duration in seconds: line 5 of the LRW .txt annotation, or the value stored in the LMDB index."""
        if self._data_format == 'lmdb':
            return self._durations[filename]
        info_txt = os.path.join(self._annonation_direc, *filename.split('/')[self.label_idx:] )  # swap base folder
        info_txt = os.path.splitext( info_txt )[0] + '.txt'   # swap extension
        info = read_txt_lines(info_txt)
        return float( info[4].split(' ')[1] )

    def load_data(self, filename):

        if self._data_format == 'lmdb':
            return self._load_data_lmdb(filename)
        try:
            if filename.endswith('npz'):
                return np.load(filename)['data']
            elif filename.endswith('mp4'):
                return librosa.load(filename, sr=16000)[0][-19456:]
            else:
                return np.load(filename)
        except IOError:
            print(f"Error when reading file: {filename}")
            sys.exit()

    def _apply_variable_length_aug(self, filename, raw_data):
        # duration of word, to be used to do temporal cropping
        utterance_duration = self._get_duration(filename)
        half_interval = int( utterance_duration/2.0 * self.fps)  # num frames of utterance / 2
                
        n_frames = raw_data.shape[0]
        mid_idx = ( n_frames -1 ) // 2  # video has n frames, mid point is (n-1)//2 as count starts with 0
        left_idx = random.randint(0, max(0,mid_idx-half_interval-1)  )   # random.randint(a,b) chooses in [a,b]
        right_idx = random.randint( min( mid_idx+half_interval+1,n_frames ), n_frames  )

        return raw_data[left_idx:right_idx]


    def _get_boundary(self, filename, raw_data):
        # duration of word, to be used to do temporal cropping
        utterance_duration = self._get_duration(filename)
        # boundary is used for the features at the top of ResNet, which as a frame rate of 25fps.
        if self.fps == 25:
            half_interval = int( utterance_duration/2.0 * self.fps)
            n_frames = raw_data.shape[0]
        elif self.fps == 16000:
            half_interval = int( utterance_duration/2.0 * 25)
            n_frames = raw_data.shape[0] // 640

        mid_idx = ( n_frames -1 ) // 2  # video has n frames, mid point is (n-1)//2 as count starts with 0
        left_idx = max(0, mid_idx-half_interval-1)
        right_idx = min(mid_idx+half_interval+1, n_frames)

        boundary = np.zeros(n_frames)
        boundary[left_idx:right_idx] = 1
        return boundary

    def __getitem__(self, idx):

        # a clip that cannot be read (empty/corrupt .npz) is skipped: the next clip is returned instead
        for _ in range(len(self)):
            try:
                raw_data = self.load_data(self.list[idx][0])
                break
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: skipping unreadable clip {self.list[idx][0]} ({type(e).__name__}: {e})", flush=True)
                idx = (idx + 1) % len(self)
        else:
            raise RuntimeError(f"No readable clip in partition {self._data_partition}")
        # -- perform variable length on training set
        if ( self._data_partition == 'train' ) and self.is_var_length and not self.use_boundary:
            data = self._apply_variable_length_aug(self.list[idx][0], raw_data)
        else:
            data = raw_data
        preprocess_data = self.preprocessing_func(data)
        label = self.list[idx][1]
        if self.use_boundary:
            boundary = self._get_boundary(self.list[idx][0], raw_data)
            return preprocess_data, label, boundary
        else:
            return preprocess_data, label

    def __len__(self):
        return len(self._data_files)


def pad_packed_collate(batch):
    if len(batch[0]) == 2:
        use_boundary = False
        data_tuple, lengths, labels_tuple = zip(*[(a, a.shape[0], b) for (a, b) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])
    elif len(batch[0]) == 3:
        use_boundary = True
        data_tuple, lengths, labels_tuple, boundaries_tuple = zip(*[(a, a.shape[0], b, c) for (a, b, c) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])

    if data_tuple[0].ndim == 1:
        max_len = data_tuple[0].shape[0]
        data_np = np.zeros((len(data_tuple), max_len))
    elif data_tuple[0].ndim == 3:
        max_len, h, w = data_tuple[0].shape
        data_np = np.zeros((len(data_tuple), max_len, h, w))
    for idx in range( len(data_np)):
        data_np[idx][:data_tuple[idx].shape[0]] = data_tuple[idx]
    data = torch.FloatTensor(data_np)

    if use_boundary:
        boundaries_np = np.zeros((len(boundaries_tuple), len(boundaries_tuple[0])))
        for idx in range(len(data_np)):
            boundaries_np[idx] = boundaries_tuple[idx]
        boundaries = torch.FloatTensor(boundaries_np).unsqueeze(-1)

    labels = torch.LongTensor(labels_tuple)

    if use_boundary:
        return data, lengths, labels, boundaries
    else:
        return data, lengths, labels
