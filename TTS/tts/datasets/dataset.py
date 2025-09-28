import base64
import collections
import os
import random
import time
import tempfile
from typing import Dict, List, Union

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from torch.utils.data import Dataset

from TTS.tts.utils.data import prepare_data, prepare_stop_target, prepare_tensor
from TTS.utils.audio import AudioProcessor
from TTS.utils.audio.numpy_transforms import compute_energy as calculate_energy

import mutagen

# Eviter "too many open files"
torch.multiprocessing.set_sharing_strategy("file_system")


# -----------------------
# DDP helpers
# -----------------------
def _is_ddp():
    return dist.is_available() and dist.is_initialized()


def _rank():
    return dist.get_rank() if _is_ddp() else 0


def _barrier():
    if _is_ddp():
        dist.barrier()


# -----------------------
# Utils
# -----------------------
def _np_load_retry(path, retries=5, delay=0.1):
    """
    np.load robuste sans pickle. Retry si lecture partielle pendant une écriture atomique.
    """
    for _ in range(retries):
        try:
            return np.load(path, allow_pickle=False)
        except FileNotFoundError:
            return None
        except Exception:
            time.sleep(delay)
    try:
        return np.load(path, allow_pickle=False)
    except Exception:
        return None


def _np_save_atomic(path, arr):
    """
    Ecriture atomique: fichier temporaire puis replace.
    """
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=d, delete=False) as tmp:
        tmp_name = tmp.name
        np.save(tmp_name, arr, allow_pickle=False)
    os.replace(tmp_name, path)


def _safe_makedirs_once(path: str):
    """Créer un dossier une seule fois à travers les ranks et synchroniser."""
    if path is None:
        return
    if _is_ddp():
        if _rank() == 0:
            os.makedirs(path, exist_ok=True)
        _barrier()
    else:
        os.makedirs(path, exist_ok=True)


def _ensure_finite_np(x, name):
    if not np.all(np.isfinite(x)):
        raise RuntimeError(f"Non-finite in {name} (numpy)")


def _ensure_finite_t(x: torch.Tensor, name):
    if not torch.isfinite(x).all():
        raise RuntimeError(f"Non-finite in {name} (torch)")


def _parse_sample(item):
    language_name = None
    attn_file = None
    if len(item) == 5:
        text, wav_file, speaker_name, language_name, attn_file = item
    elif len(item) == 4:
        text, wav_file, speaker_name, language_name = item
    elif len(item) == 3:
        text, wav_file, speaker_name = item
    else:
        raise ValueError(" [!] Dataset cannot parse the sample.")
    return text, wav_file, speaker_name, language_name, attn_file


def noise_augment_audio(wav):
    return wav + (1.0 / 32768.0) * np.random.rand(*wav.shape)


def string2filename(string):
    # nom de fichier sûr et réversible
    filename = base64.urlsafe_b64encode(string.encode("utf-8")).decode("utf-8", "ignore")
    return filename


def get_audio_size(audiopath):
    extension = audiopath.rpartition(".")[-1].lower()
    if extension not in {"mp3", "wav", "flac"}:
        raise RuntimeError(
            f"The audio format {extension} is not supported, please convert the audio files to mp3, flac, or wav format!"
        )
    audio_info = mutagen.File(audiopath).info
    return int(audio_info.length * audio_info.sample_rate)


class TTSDataset(Dataset):
    def __init__(
        self,
        outputs_per_step: int = 1,
        compute_linear_spec: bool = False,
        ap: AudioProcessor = None,
        samples: List[Dict] = None,
        tokenizer: "TTSTokenizer" = None,
        compute_f0: bool = False,
        compute_energy: bool = False,
        f0_cache_path: str = None,
        energy_cache_path: str = None,
        return_wav: bool = False,
        batch_group_size: int = 0,
        min_text_len: int = 0,
        max_text_len: int = float("inf"),
        min_audio_len: int = 0,
        max_audio_len: int = float("inf"),
        phoneme_cache_path: str = None,
        precompute_num_workers: int = 0,
        speaker_id_mapping: Dict = None,
        d_vector_mapping: Dict = None,
        language_id_mapping: Dict = None,
        use_noise_augment: bool = False,
        start_by_longest: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        self.batch_group_size = batch_group_size
        self._samples = samples
        self.outputs_per_step = outputs_per_step
        self.compute_linear_spec = compute_linear_spec
        self.return_wav = return_wav
        self.compute_f0 = compute_f0
        self.compute_energy = compute_energy
        self.f0_cache_path = f0_cache_path
        self.energy_cache_path = energy_cache_path
        self.min_audio_len = min_audio_len
        self.max_audio_len = max_audio_len
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.ap = ap
        self.phoneme_cache_path = phoneme_cache_path
        self.speaker_id_mapping = speaker_id_mapping
        self.d_vector_mapping = d_vector_mapping
        self.language_id_mapping = language_id_mapping
        self.use_noise_augment = use_noise_augment
        self.start_by_longest = start_by_longest

        self.verbose = verbose
        self.tokenizer = tokenizer

        if self.tokenizer.use_phonemes:
            self.phoneme_dataset = PhonemeDataset(
                self.samples, self.tokenizer, phoneme_cache_path, precompute_num_workers=precompute_num_workers
            )

        if compute_f0:
            self.f0_dataset = F0Dataset(
                self.samples, self.ap, cache_path=f0_cache_path, precompute_num_workers=precompute_num_workers
            )
        if compute_energy:
            self.energy_dataset = EnergyDataset(
                self.samples, self.ap, cache_path=energy_cache_path, precompute_num_workers=precompute_num_workers
            )
        if self.verbose:
            self.print_logs()

    @property
    def lengths(self):
        lens = []
        for item in self.samples:
            _, wav_file, *_ = _parse_sample(item)
            audio_len = get_audio_size(wav_file)
            lens.append(audio_len)
        return lens

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, new_samples):
        self._samples = new_samples
        if hasattr(self, "f0_dataset"):
            self.f0_dataset.samples = new_samples
        if hasattr(self, "energy_dataset"):
            self.energy_dataset.samples = new_samples
        if hasattr(self, "phoneme_dataset"):
            self.phoneme_dataset.samples = new_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.load_data(idx)

    def print_logs(self, level: int = 0) -> None:
        indent = "\t" * level
        print("\n")
        print(f"{indent}> DataLoader initialization")
        print(f"{indent}| > Tokenizer:")
        self.tokenizer.print_logs(level + 1)
        print(f"{indent}| > Number of instances : {len(self.samples)}")

    def load_wav(self, filename):
        waveform = self.ap.load_wav(filename).astype(np.float32)
        assert waveform.size > 0
        _ensure_finite_np(waveform, "waveform")
        return waveform

    def get_phonemes(self, idx, text):
        out_dict = self.phoneme_dataset[idx]
        assert text == out_dict["text"], f"{text} != {out_dict['text']}"
        assert len(out_dict["token_ids"]) > 0
        return out_dict

    def get_f0(self, idx):
        out_dict = self.f0_dataset[idx]
        item = self.samples[idx]
        assert item["audio_unique_name"] == out_dict["audio_unique_name"]
        return out_dict

    def get_energy(self, idx):
        out_dict = self.energy_dataset[idx]
        item = self.samples[idx]
        assert item["audio_unique_name"] == out_dict["audio_unique_name"]
        return out_dict

    @staticmethod
    def get_attn_mask(attn_file):
        attn = np.load(attn_file, allow_pickle=False)
        _ensure_finite_np(attn, "attn")
        return attn

    def get_token_ids(self, idx, text):
        if self.tokenizer.use_phonemes:
            token_ids = self.get_phonemes(idx, text)["token_ids"]
        else:
            token_ids = self.tokenizer.text_to_ids(text)
        token_ids = np.array(token_ids, dtype=np.int32)
        _ensure_finite_np(token_ids, "token_ids")
        return token_ids

    def load_data(self, start_idx):
        # Remplace la récursion par une boucle bornée
        tries = 0
        idx = start_idx
        n = len(self.samples)
        while tries < n:
            item = self.samples[idx]

            raw_text = item["text"]
            wav = self.load_wav(item["audio_file"])

            if self.use_noise_augment:
                wav = noise_augment_audio(wav)
                _ensure_finite_np(wav, "wav_aug")

            token_ids = self.get_token_ids(idx, item["text"])

            attn = None
            if "alignment_file" in item:
                attn = self.get_attn_mask(item["alignment_file"])

            if len(token_ids) > self.max_text_len or len(wav) < self.min_audio_len:
                tries += 1
                idx = (idx + 1) % n
                continue

            f0 = self.get_f0(idx)["f0"] if self.compute_f0 else None
            energy = self.get_energy(idx)["energy"] if self.compute_energy else None

            sample = {
                "raw_text": raw_text,
                "token_ids": token_ids,
                "wav": wav,
                "pitch": f0,
                "energy": energy,
                "attn": attn,
                "item_idx": item["audio_file"],
                "speaker_name": item["speaker_name"],
                "language_name": item["language"],
                "wav_file_name": os.path.basename(item["audio_file"]),
                "audio_unique_name": item["audio_unique_name"],
            }
            return sample

        raise RuntimeError(" [!] No valid sample found after scanning the dataset.")

    @staticmethod
    def _compute_lengths(samples):
        new_samples = []
        for item in samples:
            audio_length = get_audio_size(item["audio_file"])
            text_lenght = len(item["text"])
            item["audio_length"] = audio_length
            item["text_length"] = text_lenght
            new_samples += [item]
        return new_samples

    @staticmethod
    def filter_by_length(lengths: List[int], min_len: int, max_len: int):
        idxs = np.argsort(lengths)  # ascending order
        ignore_idx = []
        keep_idx = []
        for idx in idxs:
            length = lengths[idx]
            if length < min_len or length > max_len:
                ignore_idx.append(idx)
            else:
                keep_idx.append(idx)
        return ignore_idx, keep_idx

    @staticmethod
    def sort_by_length(samples: List[List]):
        audio_lengths = [s["audio_length"] for s in samples]
        idxs = np.argsort(audio_lengths)  # ascending order
        return idxs

    @staticmethod
    def create_buckets(samples, batch_group_size: int):
        assert batch_group_size > 0
        for i in range(len(samples) // batch_group_size):
            offset = i * batch_group_size
            end_offset = offset + batch_group_size
            temp_items = samples[offset:end_offset]
            random.shuffle(temp_items)
            samples[offset:end_offset] = temp_items
        return samples

    @staticmethod
    def _select_samples_by_idx(idxs, samples):
        samples_new = []
        for idx in idxs:
            samples_new.append(samples[idx])
        return samples_new

    def preprocess_samples(self):
        samples = self._compute_lengths(self.samples)

        text_lengths = [i["text_length"] for i in samples]
        audio_lengths = [i["audio_length"] for i in samples]
        text_ignore_idx, text_keep_idx = self.filter_by_length(text_lengths, self.min_text_len, self.max_text_len)
        audio_ignore_idx, audio_keep_idx = self.filter_by_length(audio_lengths, self.min_audio_len, self.max_audio_len)
        keep_idx = list(set(audio_keep_idx) & set(text_keep_idx))
        ignore_idx = list(set(audio_ignore_idx) | set(text_ignore_idx))

        samples = self._select_samples_by_idx(keep_idx, samples)
        sorted_idxs = self.sort_by_length(samples)

        if self.start_by_longest:
            longest_idxs = sorted_idxs[-1]
            sorted_idxs[-1] = sorted_idxs[0]
            sorted_idxs[0] = longest_idxs

        samples = self._select_samples_by_idx(sorted_idxs, samples)

        if len(samples) == 0:
            raise RuntimeError(" [!] No samples left")

        if self.batch_group_size > 0:
            samples = self.create_buckets(samples, self.batch_group_size)

        audio_lengths = [s["audio_length"] for s in samples]
        text_lengths = [s["text_length"] for s in samples]
        self.samples = samples

        if self.verbose:
            print(" | > Preprocessing samples")
            print(" | > Max text length: {}".format(np.max(text_lengths)))
            print(" | > Min text length: {}".format(np.min(text_lengths)))
            print(" | > Avg text length: {}".format(np.mean(text_lengths)))
            print(" | ")
            print(" | > Max audio length: {}".format(np.max(audio_lengths)))
            print(" | > Min audio length: {}".format(np.min(audio_lengths)))
            print(" | > Avg audio length: {}".format(np.mean(audio_lengths)))
            print(f" | > Num. instances discarded samples: {len(ignore_idx)}")
            print(" | > Batch group size: {}.".format(self.batch_group_size))

    @staticmethod
    def _sort_batch(batch, text_lengths):
        text_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor(text_lengths), dim=0, descending=True)
        batch = [batch[idx] for idx in ids_sorted_decreasing]
        return batch, text_lengths, ids_sorted_decreasing

    def collate_fn(self, batch):
        # Puts each data field into un tenseur avec dimension batch
        if isinstance(batch[0], collections.abc.Mapping):
            token_ids_lengths = np.array([len(d["token_ids"]) for d in batch], dtype=np.int64)

            # sort by text length
            batch, token_ids_lengths, ids_sorted_decreasing = self._sort_batch(batch, token_ids_lengths)

            # list[dict] -> dict[list]
            batch = {k: [dic[k] for dic in batch] for k in batch[0]}

            # mappings
            language_ids = [self.language_id_mapping[ln] for ln in batch["language_name"]] if self.language_id_mapping is not None else None
            if self.d_vector_mapping is not None:
                embedding_keys = list(batch["audio_unique_name"])
                d_vectors = [self.d_vector_mapping[w]["embedding"] for w in embedding_keys]
            else:
                d_vectors = None
            speaker_ids = [self.speaker_id_mapping[sn] for sn in batch["speaker_name"]] if self.speaker_id_mapping else None

            # features
            mel = [self.ap.melspectrogram(w).astype("float32") for w in batch["wav"]]
            for m in mel:
                _ensure_finite_np(m, "mel")
            mel_lengths = [m.shape[1] for m in mel]
            mel_lengths_adjusted = [
                m.shape[1] + (self.outputs_per_step - (m.shape[1] % self.outputs_per_step))
                if m.shape[1] % self.outputs_per_step
                else m.shape[1]
                for m in mel
            ]

            # stop targets
            stop_targets = [np.array([0.0] * (ml - 1) + [1.0], dtype=np.float32) for ml in mel_lengths]
            stop_targets = prepare_stop_target(stop_targets, self.outputs_per_step)

            token_ids = prepare_data(batch["token_ids"]).astype(np.int32)
            mel = prepare_tensor(mel, self.outputs_per_step).transpose(0, 2, 1)  # BxTxD

            token_ids_lengths = torch.LongTensor(token_ids_lengths)
            token_ids = torch.LongTensor(token_ids)
            mel = torch.FloatTensor(mel).contiguous()
            mel_lengths = torch.LongTensor(mel_lengths)
            stop_targets = torch.FloatTensor(stop_targets)

            if d_vectors is not None:
                d_vectors = torch.FloatTensor(d_vectors)
                _ensure_finite_t(d_vectors, "d_vectors")
            if speaker_ids is not None:
                speaker_ids = torch.LongTensor(speaker_ids)
            if language_ids is not None:
                language_ids = torch.LongTensor(language_ids)

            linear = None
            if self.compute_linear_spec:
                linear = [self.ap.spectrogram(w).astype("float32") for w in batch["wav"]]
                for l in linear:
                    _ensure_finite_np(l, "linear")
                linear = prepare_tensor(linear, self.outputs_per_step).transpose(0, 2, 1)
                assert mel.shape[1] == linear.shape[1]
                linear = torch.FloatTensor(linear).contiguous()

            wav_padded = None
            if self.return_wav:
                wav_lengths = [w.shape[0] for w in batch["wav"]]
                max_wav_len = max(mel_lengths_adjusted) * self.ap.hop_length
                wav_lengths = torch.LongTensor(wav_lengths)
                wav_padded = torch.zeros(len(batch["wav"]), 1, max_wav_len, dtype=torch.float32)
                for i, w in enumerate(batch["wav"]):
                    mel_length = mel_lengths_adjusted[i]
                    w = np.pad(w, (0, self.ap.hop_length * self.outputs_per_step), mode="edge")
                    w = w[: mel_length * self.ap.hop_length]
                    wav_padded[i, :, : w.shape[0]] = torch.from_numpy(w.astype(np.float32))
                wav_padded.transpose_(1, 2)

            if self.compute_f0:
                pitch = prepare_data(batch["pitch"]).astype(np.float32)
                assert mel.shape[1] == pitch.shape[1], f"[!] {mel.shape} vs {pitch.shape}"
                pitch = torch.from_numpy(pitch)[:, None, :].contiguous().to(torch.float32)
                _ensure_finite_t(pitch, "pitch_batch")
            else:
                pitch = None

            if self.compute_energy:
                energy = prepare_data(batch["energy"]).astype(np.float32)
                assert mel.shape[1] == energy.shape[1], f"[!] {mel.shape} vs {energy.shape}"
                energy = torch.from_numpy(energy)[:, None, :].contiguous().to(torch.float32)
                _ensure_finite_t(energy, "energy_batch")
            else:
                energy = None

            attns = None
            if batch["attn"][0] is not None:
                attns = [batch["attn"][idx].T for idx in ids_sorted_decreasing]
                for idx, attn in enumerate(attns):
                    pad2 = mel.shape[1] - attn.shape[1]
                    pad1 = token_ids.shape[1] - attn.shape[0]
                    assert pad1 >= 0 and pad2 >= 0, f"[!] Negative padding - {pad1} and {pad2}"
                    attn = np.pad(attn, [[0, pad1], [0, pad2]])
                    _ensure_finite_np(attn, "attn_padded")
                    attns[idx] = attn
                attns = prepare_tensor(attns, self.outputs_per_step)
                attns = torch.FloatTensor(attns).unsqueeze(1)
                # Sanity check finale des dims
                assert attns.shape[2] == token_ids.shape[1] and attns.shape[3] == mel.shape[1], \
                    f"[!] attn shape mismatch {attns.shape} vs tokens {token_ids.shape} / mel {mel.shape}"

            return {
                "token_id": token_ids,
                "token_id_lengths": token_ids_lengths,
                "speaker_names": batch["speaker_name"],
                "linear": linear,
                "mel": mel,
                "mel_lengths": mel_lengths,
                "stop_targets": stop_targets,
                "item_idxs": batch["item_idx"],
                "d_vectors": d_vectors,
                "speaker_ids": speaker_ids,
                "attns": attns,
                "waveform": wav_padded,
                "raw_text": batch["raw_text"],
                "pitch": pitch,
                "energy": energy,
                "language_ids": language_ids,
                "audio_unique_names": batch["audio_unique_name"],
            }

        raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}".format(type(batch[0]))))


class PhonemeDataset(Dataset):
    """
    DDP-safe phoneme caching:
      * rank 0 crée le dossier et peut pré-calculer
      * écriture atomique .npy
      * lecture robuste sans pickle, avec retries
    """
    def __init__(
        self,
        samples: Union[List[Dict], List[List]],
        tokenizer: "TTSTokenizer",
        cache_path: str,
        precompute_num_workers=0,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.cache_path = cache_path

        # créer dossier une fois
        need_precompute = False
        if cache_path is not None:
            if _is_ddp():
                if _rank() == 0 and not os.path.exists(cache_path):
                    os.makedirs(cache_path, exist_ok=True)
                    need_precompute = True
                _barrier()
            else:
                if not os.path.exists(cache_path):
                    os.makedirs(cache_path, exist_ok=True)
                    need_precompute = True

            # précompute seulement par rank 0, puis sync
            if need_precompute and precompute_num_workers > 0:
                if _rank() == 0:
                    self.precompute(num_workers=0)  # 0 pour éviter write concurrent
                _barrier()

    def __getitem__(self, index):
        item = self.samples[index]
        ids = self.compute_or_load(string2filename(item["audio_unique_name"]), item["text"], item["language"])
        ph_hat = self.tokenizer.ids_to_text(ids)
        return {"text": item["text"], "ph_hat": ph_hat, "token_ids": ids, "token_ids_len": len(ids)}

    def __len__(self):
        return len(self.samples)

    def compute_or_load(self, file_name, text, language):
        file_ext = "_phoneme.npy"
        cache_path = os.path.join(self.cache_path, file_name + file_ext)

        # lecture robuste si présent
        if os.path.exists(cache_path):
            ids = _np_load_retry(cache_path)
            if ids is not None:
                ids = np.asarray(ids, dtype=np.int64)
                _ensure_finite_np(ids, "phoneme_ids_cached")
                return ids

        # calcul local
        ids = self.tokenizer.text_to_ids(text, language=language)
        ids = np.asarray(ids, dtype=np.int64)
        _ensure_finite_np(ids, "phoneme_ids_new")

        # seul rank 0 écrit (atomique)
        if _rank() == 0:
            try:
                _np_save_atomic(cache_path, ids)
            except Exception:
                try:
                    if os.path.exists(cache_path):
                        os.remove(cache_path)
                except Exception:
                    pass
        return ids

    def get_pad_id(self):
        return self.tokenizer.pad_id

    def precompute(self, num_workers=0):
        print("[*] Pre-computing phonemes...")
        with tqdm.tqdm(total=len(self)) as pbar:
            batch_size = 1
            dataloder = torch.utils.data.DataLoader(
                batch_size=batch_size, dataset=self, shuffle=False, num_workers=num_workers, collate_fn=self.collate_fn
            )
            for _ in dataloder:
                pbar.update(batch_size)

    def collate_fn(self, batch):
        ids = [item["token_ids"] for item in batch]
        ids_lens = [item["token_ids_len"] for item in batch]
        texts = [item["text"] for item in batch]
        texts_hat = [item["ph_hat"] for item in batch]
        ids_lens_max = max(ids_lens)
        ids_torch = torch.LongTensor(len(ids), ids_lens_max).fill_(self.get_pad_id())
        for i, ids_len in enumerate(ids_lens):
            ids_torch[i, :ids_len] = torch.LongTensor(ids[i])
        return {"text": texts, "ph_hat": texts_hat, "token_ids": ids_torch}

    def print_logs(self, level: int = 0) -> None:
        indent = "\t" * level
        print("\n")
        print(f"{indent}> PhonemeDataset ")
        print(f"{indent}| > Tokenizer:")
        self.tokenizer.print_logs(level + 1)
        print(f"{indent}| > Number of instances : {len(self.samples)}")


class F0Dataset:
    """DDP-safe F0 cache (single writer + atomic save)."""
    def __init__(
        self,
        samples: Union[List[List], List[Dict]],
        ap: "AudioProcessor",
        audio_config=None,  # pylint: disable=unused-argument
        verbose=False,
        cache_path: str = None,
        precompute_num_workers=0,
        normalize_f0=True,
    ):
        self.samples = samples
        self.ap = ap
        self.verbose = verbose
        self.cache_path = cache_path
        self.normalize_f0 = normalize_f0
        self.pad_id = 0.0
        self.mean = None
        self.std = None

        need_precompute = False
        if cache_path is not None:
            if _is_ddp():
                if _rank() == 0 and not os.path.exists(cache_path):
                    os.makedirs(cache_path, exist_ok=True)
                    need_precompute = True
                _barrier()
            else:
                if not os.path.exists(cache_path):
                    os.makedirs(cache_path, exist_ok=True)
                    need_precompute = True

            if need_precompute and precompute_num_workers > 0:
                if _rank() == 0:
                    self.precompute(num_workers=0)
                _barrier()

        if normalize_f0:
            self.load_stats(cache_path)

    def __getitem__(self, idx):
        item = self.samples[idx]
        f0 = self.compute_or_load(item["audio_file"], string2filename(item["audio_unique_name"]))
        if self.normalize_f0:
            assert self.mean is not None and self.std is not None, " [!] Mean and STD is not available"
            f0 = self.normalize(f0)
        return {"audio_unique_name": item["audio_unique_name"], "f0": f0}

    def __len__(self):
        return len(self.samples)

    def precompute(self, num_workers=0):
        print("[*] Pre-computing F0s...")
        with tqdm.tqdm(total=len(self)) as pbar:
            batch_size = 1
            normalize_f0 = self.normalize_f0
            self.normalize_f0 = False
            dataloder = torch.utils.data.DataLoader(
                batch_size=batch_size, dataset=self, shuffle=False, num_workers=num_workers, collate_fn=self.collate_fn
            )
            computed_data = []
            for batch in dataloder:
                f0 = batch["f0"]
                computed_data.extend(f0)  # corrige append de générateur
                pbar.update(batch_size)
            self.normalize_f0 = normalize_f0

        if self.normalize_f0:
            pitch_mean, pitch_std = self.compute_pitch_stats(computed_data)
            pitch_stats = {"mean": pitch_mean, "std": pitch_std}
            _np_save_atomic(os.path.join(self.cache_path, "pitch_stats.npy"), pitch_stats)

    def get_pad_id(self):
        return self.pad_id

    @staticmethod
    def create_pitch_file_path(file_name, cache_path):
        return os.path.join(cache_path, file_name + "_pitch.npy")

    @staticmethod
    def _compute_and_save_pitch(ap, wav_file, pitch_file=None):
        wav = ap.load_wav(wav_file).astype(np.float32)
        _ensure_finite_np(wav, "f0_wav")
        pitch = ap.compute_f0(wav).astype(np.float32)
        _ensure_finite_np(pitch, "f0_values")
        if pitch_file:
            _np_save_atomic(pitch_file, pitch)
        return pitch

    @staticmethod
    def compute_pitch_stats(pitch_vecs):
        nonzeros = np.concatenate([v[np.where(v != 0.0)[0]] for v in pitch_vecs]) if len(pitch_vecs) > 0 else np.array([1.0], dtype=np.float32)
        mean, std = np.mean(nonzeros), np.std(nonzeros)
        return np.float32(mean), np.float32(std if std > 1e-8 else 1.0)

    def load_stats(self, cache_path):
        stats_path = os.path.join(cache_path, "pitch_stats.npy")
        stats = _np_load_retry(stats_path)
        if stats is None:
            return
        stats = stats.item()
        self.mean = np.float32(stats["mean"])
        self.std = np.float32(stats["std"] if stats["std"] > 1e-8 else 1.0)

    def normalize(self, pitch):
        zero_idxs = np.where(pitch == 0.0)[0]
        pitch = pitch - self.mean
        pitch = pitch / self.std
        pitch[zero_idxs] = 0.0
        return pitch.astype(np.float32)

    def denormalize(self, pitch):
        zero_idxs = np.where(pitch == 0.0)[0]
        pitch = pitch * self.std
        pitch = pitch + self.mean
        pitch[zero_idxs] = 0.0
        return pitch.astype(np.float32)

    def compute_or_load(self, wav_file, audio_unique_name):
        pitch_file = self.create_pitch_file_path(audio_unique_name, self.cache_path)
        if os.path.exists(pitch_file):
            pitch = _np_load_retry(pitch_file)
            if pitch is not None:
                pitch = pitch.astype(np.float32)
                _ensure_finite_np(pitch, "f0_cached")
                return pitch

        pitch = self._compute_and_save_pitch(self.ap, wav_file, pitch_file if _rank() == 0 else None)
        return pitch.astype(np.float32)

    def collate_fn(self, batch):
        audio_unique_name = [item["audio_unique_name"] for item in batch]
        f0s = [item["f0"] for item in batch]
        f0_lens = [len(item["f0"]) for item in batch]
        f0_lens_max = max(f0_lens)
        # FloatTensor, pas LongTensor
        f0s_torch = torch.full((len(f0s), f0_lens_max), fill_value=self.get_pad_id(), dtype=torch.float32)
        for i, f0_len in enumerate(f0_lens):
            f0s_torch[i, :f0_len] = torch.from_numpy(f0s[i].astype(np.float32))
        return {"audio_unique_name": audio_unique_name, "f0": f0s_torch, "f0_lens": f0_lens}

    def print_logs(self, level: int = 0) -> None:
        indent = "\t" * level
        print("\n")
        print(f"{indent}> F0Dataset ")
        print(f"{indent}| > Number of instances : {len(self.samples)}")


class EnergyDataset:
    """DDP-safe Energy cache (single writer + atomic save)."""
    def __init__(
        self,
        samples: Union[List[List], List[Dict]],
        ap: "AudioProcessor",
        verbose=False,
        cache_path: str = None,
        precompute_num_workers=0,
        normalize_energy=True,
    ):
        self.samples = samples
        self.ap = ap
        self.verbose = verbose
        self.cache_path = cache_path
        self.normalize_energy = normalize_energy
        self.pad_id = 0.0
        self.mean = None
        self.std = None

        need_precompute = False
        if cache_path is not None:
            if _is_ddp():
                if _rank() == 0 and not os.path.exists(cache_path):
                    os.makedirs(cache_path, exist_ok=True)
                    need_precompute = True
                _barrier()
            else:
                if not os.path.exists(cache_path):
                    os.makedirs(cache_path, exist_ok=True)
                    need_precompute = True

            if need_precompute and precompute_num_workers > 0:
                if _rank() == 0:
                    self.precompute(num_workers=0)
                _barrier()

        if normalize_energy:
            self.load_stats(cache_path)

    def __getitem__(self, idx):
        item = self.samples[idx]
        energy = self.compute_or_load(item["audio_file"], string2filename(item["audio_unique_name"]))
        if self.normalize_energy:
            assert self.mean is not None and self.std is not None, " [!] Mean and STD is not available"
            energy = self.normalize(energy)
        return {"audio_unique_name": item["audio_unique_name"], "energy": energy}

    def __len__(self):
        return len(self.samples)

    def precompute(self, num_workers=0):
        print("[*] Pre-computing energys...]")
        with tqdm.tqdm(total=len(self)) as pbar:
            batch_size = 1
            normalize_energy = self.normalize_energy
            self.normalize_energy = False
            dataloder = torch.utils.data.DataLoader(
                batch_size=batch_size, dataset=self, shuffle=False, num_workers=num_workers, collate_fn=self.collate_fn
            )
            computed_data = []
            for batch in dataloder:
                energy = batch["energy"]
                computed_data.extend(energy)  # corrige append de générateur
                pbar.update(batch_size)
            self.normalize_energy = normalize_energy

        if self.normalize_energy:
            energy_mean, energy_std = self.compute_energy_stats(computed_data)
            energy_stats = {"mean": energy_mean, "std": energy_std}
            _np_save_atomic(os.path.join(self.cache_path, "energy_stats.npy"), energy_stats)

    def get_pad_id(self):
        return self.pad_id

    @staticmethod
    def create_energy_file_path(wav_or_name, cache_path):
        # compat: accepte chemin complet ou nom unique
        base = os.path.splitext(os.path.basename(wav_or_name))[0]
        return os.path.join(cache_path, base + "_energy.npy")

    @staticmethod
    def _compute_and_save_energy(ap, wav_file, energy_file=None):
        wav = ap.load_wav(wav_file).astype(np.float32)
        _ensure_finite_np(wav, "energy_wav")
        energy = calculate_energy(wav, fft_size=ap.fft_size, hop_length=ap.hop_length, win_length=ap.win_length).astype(
            np.float32
        )
        _ensure_finite_np(energy, "energy_values")
        if energy_file:
            _np_save_atomic(energy_file, energy)
        return energy

    @staticmethod
    def compute_energy_stats(energy_vecs):
        nonzeros = np.concatenate([v[np.where(v != 0.0)[0]] for v in energy_vecs]) if len(energy_vecs) > 0 else np.array([1.0], dtype=np.float32)
        mean, std = np.mean(nonzeros), np.std(nonzeros)
        return np.float32(mean), np.float32(std if std > 1e-8 else 1.0)

    def load_stats(self, cache_path):
        stats_path = os.path.join(cache_path, "energy_stats.npy")
        stats = _np_load_retry(stats_path)
        if stats is None:
            return
        stats = stats.item()
        self.mean = np.float32(stats["mean"])
        self.std = np.float32(stats["std"] if stats["std"] > 1e-8 else 1.0)

    def normalize(self, energy):
        zero_idxs = np.where(energy == 0.0)[0]
        energy = energy - self.mean
        energy = energy / self.std
        energy[zero_idxs] = 0.0
        return energy.astype(np.float32)

    def denormalize(self, energy):
        zero_idxs = np.where(energy == 0.0)[0]
        energy = energy * self.std
        energy = energy + self.mean
        energy[zero_idxs] = 0.0
        return energy.astype(np.float32)

    def compute_or_load(self, wav_file, audio_unique_name):
        energy_file = self.create_energy_file_path(audio_unique_name, self.cache_path)
        if os.path.exists(energy_file):
            energy = _np_load_retry(energy_file)
            if energy is not None:
                energy = energy.astype(np.float32)
                _ensure_finite_np(energy, "energy_cached")
                return energy

        energy = self._compute_and_save_energy(self.ap, wav_file, energy_file if _rank() == 0 else None)
        return energy.astype(np.float32)

    def collate_fn(self, batch):
        audio_unique_name = [item["audio_unique_name"] for item in batch]
        energys = [item["energy"] for item in batch]
        energy_lens = [len(item["energy"]) for item in batch]
        energy_lens_max = max(energy_lens)
        # FloatTensor, pas LongTensor
        energys_torch = torch.full((len(energys), energy_lens_max), fill_value=self.get_pad_id(), dtype=torch.float32)
        for i, energy_len in enumerate(energy_lens):
            energys_torch[i, :energy_len] = torch.from_numpy(energys[i].astype(np.float32))
        return {"audio_unique_name": audio_unique_name, "energy": energys_torch, "energy_lens": energy_lens}

    def print_logs(self, level: int = 0) -> None:
        indent = "\t" * level
        print("\n")
        print(f"{indent}> energyDataset ")
        print(f"{indent}| > Number of instances : {len(self.samples)}")
