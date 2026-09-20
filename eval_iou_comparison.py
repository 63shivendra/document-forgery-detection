import json, cv2, os, torch
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from src.models.forgery_net import ForgeryNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models
m_old = ForgeryNet(out_channels=1, encoder_name='efficientnet-b2').to(device)
m_new = ForgeryNet(out_channels=1, encoder_name='efficientnet-b2').to(device)

p_old = r'd:\shivendra pratap singh\onwardsmrechant\project final\merchant_fina\bigpower\best_model_splicing_combined_efficientnet-b2_20260908_1523.pth'
p_new = r'd:\shivendra pratap singh\onwardsmrechant\project final\merchant_fina\bigpower\best_model_indian_kyc_enhanced_20260917_1601.pth'

ck_old = torch.load(p_old, map_location=device)
m_old.load_state_dict(ck_old.get('model_state_dict', ck_old))
m_old.eval()

ck_new = torch.load(p_new, map_location=device)
m_new.load_state_dict(ck_new.get('model_state_dict', ck_new))
m_new.eval()

with open(r'd:\shivendra pratap singh\onwardsmrechant\project final\merchant_fina\testing_live\manifest.json', 'r') as f:
    manifest = json.load(f)['dataset_manifest']

tamp = [s for s in manifest if s['is_tampered']]

ious_old, ious_new = [], []
by_type = {}

for s in tamp:
    ft = s['forgery_type']
    if ft not in by_type:
        by_type[ft] = {'old': [], 'new': []}
        
    img_p = os.path.join(r'd:\shivendra pratap singh\onwardsmrechant\project final\merchant_fina\testing_live\images', s['image_filename'])
    msk_p = os.path.join(r'd:\shivendra pratap singh\onwardsmrechant\project final\merchant_fina\testing_live\masks', s['mask_filename'])
    
    gt = cv2.imread(msk_p, 0)
    gt_bin = (cv2.resize(gt, (384, 384)) > 127).astype(np.uint8)
    
    img = Image.open(img_p).convert('RGB').resize((384, 384), Image.BILINEAR)
    t = TF.normalize(TF.to_tensor(img), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pr_old = (torch.sigmoid(m_old(t)).squeeze().cpu().numpy() >= 0.50).astype(np.uint8)
        pr_new = (torch.sigmoid(m_new(t)).squeeze().cpu().numpy() >= 0.50).astype(np.uint8)
        
    inter_o = (pr_old & gt_bin).sum()
    union_o = (pr_old | gt_bin).sum()
    iou_o = (inter_o / float(union_o)) * 100.0 if union_o > 0 else 0.0
    
    inter_n = (pr_new & gt_bin).sum()
    union_n = (pr_new | gt_bin).sum()
    iou_n = (inter_n / float(union_n)) * 100.0 if union_n > 0 else 0.0
    
    ious_old.append(iou_o)
    ious_new.append(iou_n)
    by_type[ft]['old'].append(iou_o)
    by_type[ft]['new'].append(iou_n)

print('=' * 65)
print('  EXACT MASK LOCALIZATION IoU COMPARISON (60 FORGED SAMPLES)')
print('=' * 65)
print(f'Overall Mean IoU:')
print(f'  Previous Base Model : {np.mean(ious_old):.2f}%')
print(f'  New Enhanced Model  : {np.mean(ious_new):.2f}%')
print(f'  Net IoU Improvement : +{np.mean(ious_new) - np.mean(ious_old):.2f}%')
print('-' * 65)
print('Per-Forgery-Type Mean IoU:')
for ft, vals in sorted(by_type.items()):
    mean_o = np.mean(vals['old'])
    mean_n = np.mean(vals['new'])
    print(f'  - {ft:20s}: Base={mean_o:5.2f}% -> New={mean_n:5.2f}% (+{mean_n - mean_o:5.2f}%)')
print('=' * 65)
