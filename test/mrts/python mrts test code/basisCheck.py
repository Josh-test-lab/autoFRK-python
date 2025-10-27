import numpy as np
from frk_basis import *
from mrts import *

locations = np.loadtxt('loc.csv', delimiter=',', skiprows=1)  # skiprows=1 跳過標題
print(locations)
print(locations.shape)

locs = torch.tensor(locations, dtype=torch.float32)

frk_basis = FRKBasis(locs, 20, device='cpu')
frk_basis = MRTS(locs, 20, device='cpu')
F, BBBH, UZ, eigenvalues = frk_basis()
F_df = pd.DataFrame(F.cpu().detach().numpy())
BBBH_df = pd.DataFrame(BBBH.cpu().detach().numpy())
UZ_df = pd.DataFrame(UZ)

# 匯出 CSV
F_df.to_csv('F_output.csv', index=False, encoding='utf-8-sig')
BBBH_df.to_csv('BBBH_output.csv', index=False, encoding='utf-8-sig')
UZ_df.to_csv('UZ_output.csv', index=False, encoding='utf-8-sig')