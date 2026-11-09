# Run this CELL in Google Colab after mounting Drive.
# It creates one ZIP of notebook/results/figures and excludes model weight files.
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from google.colab import drive, files

drive.mount('/content/drive')
root = Path('/content/drive/MyDrive/GenAI_Assignment_2')
out = Path('/content/GenAI_A2_colab_artifacts.zip')
allowed = {'.ipynb', '.json', '.jsonl', '.csv', '.png', '.jpg', '.jpeg', '.svg', '.txt', '.md', '.pdf'}
with ZipFile(out, 'w', ZIP_DEFLATED) as z:
    for folder in ('notebooks', 'results', 'figures'):
        for f in (root / folder).rglob('*') if (root / folder).exists() else []:
            if f.is_file() and f.suffix.lower() in allowed:
                z.write(f, f.relative_to(root))
print('Created', out, out.stat().st_size, 'bytes')
files.download(str(out))
# If your .ipynb is elsewhere in Drive, download it from Colab File > Download .ipynb
# and place it in the local notebooks/ directory.
