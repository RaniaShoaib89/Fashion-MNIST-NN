
---

## Requirements

- Python 3.12
- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- torchvision

---

## How to Reproduce

### Option 1 — Run on Kaggle (Recommended)

This notebook was built and run on Kaggle. To reproduce exactly:

1. Go to [Kaggle](https://www.kaggle.com) and sign in
2. Click **Create → New Notebook**
3. Upload `assignment-1.ipynb` via **File → Import Notebook**
4. Attach the dataset:
   - Click **Add Data** (right panel)
   - Search: `zalando-research/fashionmnist`
   - Click **Add**
5. Enable GPU:
   - Right panel → **Session options** → **Accelerator** → **GPU T4 x2**
6. Click **Run All** (▶▶ button at top)
7. Full runtime: approximately 15–20 minutes on GPU

### Option 2 — Run Locally

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

2. Install dependencies:
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

3. Download the Fashion-MNIST dataset:
```python
from torchvision import datasets
datasets.FashionMNIST(root='./data', train=True, download=True)
datasets.FashionMNIST(root='./data', train=False, download=True)
```

4. Update the data paths in the notebook — replace:
```python
"/kaggle/input/datasets/zalando-research/fashionmnist/fashion-mnist_train.csv"
```
with:
```python
"./data/FashionMNIST/raw/fashion-mnist_train.csv"
```

5. Open and run the notebook:
```bash
jupyter notebook assignment-1.ipynb
```

---

## Reproducibility

- Random seed fixed at `SEED = 42` throughout
- `torch.manual_seed(42)` and `np.random.seed(42)` set at the start of each part
- Stratified splits used for train/val and k-fold CV
- Small result variations (±0.5%) are expected between CPU and GPU runs

---

## Key Findings

- **Adam** converges fastest, reaching 85% validation accuracy in just 2 epochs
- **Dropout (p=0.4)** was the most effective regulariser, reducing the
  generalisation gap from 16.43% to 9.40%
- **Batch normalisation** hurt performance in the overfit setting, worsening
  the gap to 19.77%
- **More training data (n=20,000)** gave the highest validation accuracy (87.67%)
- The final tuned model achieved **89.78% test accuracy** with balanced
  macro precision, recall and F1 of ~89.75%, a +0.97pp improvement over baseline
