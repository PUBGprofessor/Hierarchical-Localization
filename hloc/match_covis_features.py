"""Run HLoc matchers on keypoints inside pair-specific covisibility masks.

The output ``matches0`` is expanded back to the original query feature indices
and its values reference original map feature indices, so localize_sfm and the
triangulated reference model remain compatible with the unmodified feature H5s.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from . import matchers
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair, parse_retrieval


def main(conf, pairs, features_query, features_map, matches, mask_provider, overwrite=False):
    pairs = Path(pairs)
    matches = Path(matches)
    matches.parent.mkdir(parents=True, exist_ok=True)
    retrieval = parse_retrieval(pairs)
    image_pairs = [(q, r) for q, refs in retrieval.items() for r in refs]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(matchers, conf["model"]["name"])
    model = Model(conf["model"]).eval().to(device)
    with h5py.File(features_query, "r") as fq, h5py.File(features_map, "r") as fm, h5py.File(matches, "a", libver="latest") as out:
        for qname, mname in tqdm(image_pairs, desc="masked feature matching"):
            pair_key = names_to_pair(qname, mname)
            if not overwrite and pair_key in out:
                continue
            qgroup, mgroup = fq[qname], fm[mname]
            qmask, mmask = mask_provider(qname, mname)
            qxy = np.asarray(qgroup["keypoints"])
            mxy = np.asarray(mgroup["keypoints"])
            qi = np.flatnonzero((qxy[:, 0] >= 0) & (qxy[:, 0] < qmask.shape[1]) & (qxy[:, 1] >= 0) & (qxy[:, 1] < qmask.shape[0]) & qmask[np.clip(qxy[:, 1].astype(int), 0, qmask.shape[0]-1), np.clip(qxy[:, 0].astype(int), 0, qmask.shape[1]-1)])
            mi = np.flatnonzero((mxy[:, 0] >= 0) & (mxy[:, 0] < mmask.shape[1]) & (mxy[:, 1] >= 0) & (mxy[:, 1] < mmask.shape[0]) & mmask[np.clip(mxy[:, 1].astype(int), 0, mmask.shape[0]-1), np.clip(mxy[:, 0].astype(int), 0, mmask.shape[1]-1)])
            if not len(qi) or not len(mi):
                expanded = np.full(len(qxy), -1, dtype=np.int16)
                scores = np.zeros(len(qxy), dtype=np.float16)
            else:
                data = {}
                for suffix, group, indices in (("0", qgroup, qi), ("1", mgroup, mi)):
                    data[f"keypoints{suffix}"] = torch.from_numpy(np.asarray(group["keypoints"])[indices][None].astype(np.float32)).to(device)
                    desc = np.asarray(group["descriptors"])
                    # HLoc SuperPoint stores descriptors as [D, N]; LightGlue's
                    # HLoc wrapper transposes the last two axes before inference.
                    if desc.shape[0] == 256:
                        desc = desc[:, indices]
                    elif desc.shape[1] == 256:
                        desc = desc[indices].T
                    else:
                        raise ValueError(f"Unexpected descriptor shape {desc.shape} in {suffix} for {qname}, {mname}")
                    data[f"descriptors{suffix}"] = torch.from_numpy(desc[None].astype(np.float32)).to(device)
                    size = np.asarray(group["image_size"], dtype=np.float32)
                    data[f"image{suffix}"] = torch.empty((1, 1, int(size[1]), int(size[0])), device=device)
                with torch.no_grad():
                    pred = model(data)
                m0 = pred["matches0"][0].detach().cpu().numpy().astype(np.int64)
                expanded = np.full(len(qxy), -1, dtype=np.int16)
                valid = m0 >= 0
                expanded[qi[valid]] = mi[m0[valid]].astype(np.int16)
                scores = np.zeros(len(qxy), dtype=np.float16)
                if "matching_scores0" in pred:
                    scores[qi] = pred["matching_scores0"][0].detach().cpu().numpy().astype(np.float16)
            if pair_key in out:
                del out[pair_key]
            grp = out.create_group(pair_key)
            grp.create_dataset("matches0", data=expanded)
            grp.create_dataset("matching_scores0", data=scores)
    return matches
