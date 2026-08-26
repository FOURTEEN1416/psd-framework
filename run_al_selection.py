"""Quick uncertainty sampling on unified pool, produce top-100/200 lists."""
import sys, os, json, pickle, numpy as np, torch, datetime
sys.path.insert(0, '.')
from psd.models.stgcn_bc import build_stgcn_bc
from psd.training.active_learning import entropy_scores, softmax_np

# Load pool
with open('runs/data_campaign/unified/real_expansion_pool_v1.pkl', 'rb') as f:
    pool = pickle.load(f, encoding='latin1')
entries = pool['entries']
print('entries count:', len(entries))

# Inspect first entry fields
print('first entry keys:', list(entries[0].keys()))
sample = entries[0]
print('sample_id:', sample.get('sample_id'))
print('source_channel:', sample.get('source_channel'))
print('label:', sample.get('label'))
print('psd_class:', sample.get('psd_class'))

# Build model using config parameters (matching p05_al_short.yaml)
model = build_stgcn_bc(in_channels=3, num_classes=22, base_channels=64, num_stages=10)
ckpt = torch.load('runs/p05_stgcn_bc_full/best.pt', map_location='cpu', weights_only=False)
# load, ignoring strict size mismatch for older checkpoints; we'll load strict=False and take matching keys
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()
device = 'cpu'
print('model loaded (strict=False)')

# Helper to convert keypoints to tensor (assuming shape (T,24,3) numpy)
def kpt_to_tensor(kp_array):
    arr = np.asarray(kp_array, dtype=np.float32)
    # if already (T,24,3) fine, else squeeze
    if arr.ndim == 3 and arr.shape[1] == 24 and arr.shape[2] == 3:
        pass
    elif arr.ndim == 2:
        # maybe (24*3) flatten? skip
        arr = arr.reshape(-1, 24, 3)
    return torch.from_numpy(arr).float()

# Run on first 200 entries for speed
sample_size = min(200, len(entries))
results = []
for i in range(sample_size):
    e = entries[i]
    kp = e.get('keypoints')
    if kp is None:
        continue
    try:
        kpt_tensor = kpt_to_tensor(kp).unsqueeze(0).to(device)
        logits = model(kpt_tensor)[0].detach().cpu().numpy()  # (C,)
        probs = softmax_np(logits[np.newaxis, :])[0]  # (C,)
        ent = float(entropy_scores(probs[np.newaxis, :])[0])
        # margin: difference between top two logits
        top_idx = np.argsort(-logits)[:2]
        top1 = logits[top_idx[0]]
        top2 = logits[top_idx[1]]
        margin = float(top1 - top2)
        results.append({
            'sample_id': e.get('sample_id'),
            'source_channel': e.get('source_channel'),
            'entropy': ent,
            'margin': margin,
            'pseudo_label': e.get('psd_class'),
            'confidence': float(probs.max()),
        })
    except Exception as ex:
        print('error entry', i, ex)
        continue

# Sort by entropy descending
results_ent = sorted(results, key=lambda x: x['entropy'], reverse=True)
print('Top 5 by entropy:')
for r in results_ent[:5]:
    print(r)

# Sort by margin ascending (low margin = uncertain)
results_margin = sorted(results, key=lambda x: x['margin'])
print('Top 5 by low margin:')
for r in results_margin[:5]:
    print(r)

# Produce top-100 and top-200 based on entropy
topk_names = ['topk_entropy_100', 'topk_entropy_200']
for kk in ['topk_entropy_100', 'topk_entropy_200']:
    kk_list = results_ent[:100] if kk == 'topk_entropy_100' else (results_ent[:200] if len(results_ent) >= 200 else results_ent)
    # keep only needed fields
    out_list = [{k: v for k, v in r.items() if k in ('sample_id','source_channel','entropy','margin','pseudo_label','confidence')} for r in kk_list]
    out_path = f'reports/al-selection-{kk}-{datetime.datetime.now().strftime("%Y%m%d")}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({kk: out_list}, f, ensure_ascii=False, indent=2)
    print(f'saved {out_path}')

# Distribution report
from collections import Counter
ch_counter = Counter(r['source_channel'] for r in results if r.get('source_channel'))
lab_counter = Counter(r['pseudo_label'] for r in results if r.get('pseudo_label'))
report = {
    'source_channel_dist': dict(ch_counter),
    'label_dist': dict(lab_counter),
    'n_scored': len(results),
}
dist_path = f'reports/al-selection-dist-{datetime.datetime.now().strftime("%Y%m%d")}.json'
with open(dist_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print('saved', dist_path)
print('Done.')