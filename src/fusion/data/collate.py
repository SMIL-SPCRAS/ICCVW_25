from collections import defaultdict
import torch

def multimodal_collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, any]]]) \
    -> list[tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, any]]]:
    """
    Collate function that merges variable modality inputs into batched tensors.
    Each item in batch: (features: dict[str, Tensor], targets: dict[str, Tensor], meta: dict)
    """
    all_modalities = set()
    for sample in batch:
        all_modalities.update(sample[0].keys())

    batch_size = len(batch)
    feature_dims = {}
    for sample in batch:
        for mod, feat in sample[0].items():
            if mod not in feature_dims:
                feature_dims[mod] = feat.shape[-1]

    features = {mod: [] for mod in all_modalities}
    targets = {"emo": []}
    metas = []

    for feat_dict, target_dict, meta in batch:
        for mod in all_modalities:
            if mod in feat_dict:
                features[mod].append(feat_dict[mod])
            else:
                features[mod].append(torch.zeros(feature_dims[mod]))
        
        targets["emo"].append(target_dict["emo"])
        metas.append(meta)

    batched_features = {mod: torch.stack(feats) for mod, feats in features.items()}
    batched_targets = {"emo": torch.stack(targets["emo"])}
    batched_metas = defaultdict(list)
    for meta in metas:
        for k, v in meta.items():
            batched_metas[k].append(v)
        
    return batched_features, batched_targets, batched_metas