Link to data: https://drive.google.com/drive/folders/1XV6l5pUOTKL129gkRrS7mvr1E3q4-mjI?usp=sharing
=======
# Installation
!pip install \
deepspeed==0.9.2 \
ffmpeg-python==0.2.0 \
more_itertools==9.1.0 \
numpy==1.23.0 \
pandas==1.4.2 \
pytorch_lightning==2.0.2 \
PyYAML==6.0 \
regex==2022.10.31 \
sacrebleu==2.3.1 \
torch==1.13.1 \
torchaudio==0.13.1 \
torchmetrics==0.11.4 \
tqdm==4.64.0 \
transformers==4.29.1

!pip install transformers==4.29.1 pytorch_lightning==2.0.2 deepspeed==0.9.2 ffmpeg-python more_itertools==9.1.0 PyYAML==6.0 regex==2022.10.31 sacrebleu==2.3.1 torchmetrics==0.11.4 tqdm==4.64.0


!pip install numpy==1.26.4 --force-reinstall

# Create folder ouput
```
import os

output_path = "/kaggle/working/comsl/output/comsl_mn2en/logs"
os.makedirs(output_path, exist_ok=True)
print(f"Đã tạo thư mục: {output_path}")

```

# Fix path audio_root + data_root
file_path = "data/data_util.py"
with open(file_path, "r") as f:
    code = f.read()

# Replace 'audio_root' to true path 
```
import re
code = re.sub(
    r"os\.path\.join\(data_root, ?'mn', ?'clips'\)",
    "os.path.join(data_root, 'extracted', 'extracted', 'mn', 'clips')",
    code,
)
code = re.sub(
    r"os\.path\.join\(data_root, ?'extracted', ?data_lang_code, ?'clips'\)",
    "os.path.join(data_root, 'extracted', 'extracted', data_lang_code, 'clips')",
    code,
)
with open(file_path, "w") as f:
    f.write(code)

print("Đã sửa audio_root cho đúng cấu trúc Kaggle!")
```

# divide train/test/dev
## train
```
import pandas as pd

# Đọc file gốc (tsv)
tsv_path = "/kaggle/input/dataset-mongolia/covost_v2.mn_en.train.tsv"
df = pd.read_csv(tsv_path, sep="\t")

# Giữ lại 1200 dòng đầu tiên
df_small = df.head(1200)

# Lưu ra file mới (ví dụ ở thư mục working, hoặc ghi đè nếu muốn)
small_tsv_path = "/kaggle/working/covost_v2.mn_en.train.1200.tsv"
df_small.to_csv(small_tsv_path, sep="\t", index=False)

print(f"Đã tạo file: {small_tsv_path} với {len(df_small)} dòng")
```
## test
```
import pandas as pd

# Đọc file gốc (tsv)
tsv_path = "/kaggle/input/dataset-mongolia/covost_v2.mn_en.test.tsv"
df = pd.read_csv(tsv_path, sep="\t")

# Giữ lại 150 dòng đầu tiên
df_small = df.head(150)

# Lưu ra file mới (ví dụ ở thư mục working, hoặc ghi đè nếu muốn)
small_tsv_path = "/kaggle/working/covost_v2.mn_en.test.1200.tsv"
df_small.to_csv(small_tsv_path, sep="\t", index=False)

print(f"Đã tạo file: {small_tsv_path} với {len(df_small)} dòng")
```

## dev
```
import pandas as pd

# Đọc file gốc (tsv)
tsv_path = "/kaggle/input/dataset-mongolia/covost_v2.mn_en.dev.tsv"
df = pd.read_csv(tsv_path, sep="\t")

# Giữ lại 150 dòng đầu tiên
df_small = df.head(150)

# Lưu ra file mới (ví dụ ở thư mục working, hoặc ghi đè nếu muốn)
small_tsv_path = "/kaggle/working/covost_v2.mn_en.dev.1200.tsv"
df_small.to_csv(small_tsv_path, sep="\t", index=False)

print(f"Đã tạo file: {small_tsv_path} với {len(df_small)} dòng")
```

# Run
```
%cd /kaggle/working/comsl
!python /kaggle/working/comsl/run.py --config comsl.yaml
```