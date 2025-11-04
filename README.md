# Predicting m6A probabilities on cancer cell lines

This README teaches users how to predict m6A probabilities on cancer cell lines using the pre-trained Random Forest model and one-hot encoder provided. Follow the steps in order.

---

## 1. Machine specification (Ubuntu)

Ensure your Ubuntu machine is set to the following specs:

- **T3.2XLarge (32Gb RAM, 8 vCPUs)**

  This can be chosen by clicking the **pencil** icon on your machine type ➜ **General Purpose Machine** ➜ scroll down to **T3** ➜ select **T3.2XLarge**.

- **50Gb SSD**

---

## 2. Folder & initial data download (assumes you're at your home directory after connecting to the instance)

> (Lecture / week 7 instructions already cover how to connect to the instance.)

Once connected, run the following commands in sequence:

```bash
mkdir helixhomies
cd helixhomies
wget http://sg-nex-data.s3-website-ap-southeast1.amazonaws.com/data/processed_data/m6Anet/SGNex_MCF7_directRNA_replicate3_run1/data.json
# (edit the URL above to the specific json file you're asked to download)
```

---

## 3. Transfer `ohe.pkl` and `random_forest_model.pkl` from GitHub/local machine

Download the `ohe.pkl` and `random_forest_model.pkl` (from our GitHub) to the same folder as your `.pem` file on your local machine. These two `.pkl` files contain stored artifacts so no retraining is required on dataset0.

**Workflow (disconnect, SCP, reconnect):**

1. Exit the instance:

```bash
exit
```

2. From your local machine (where the `.pem` file and the two `.pkl` files are), run the `scp` command. Edit `YourMachineAddress` to your machine's address and adjust path separators as appropriate for your OS.

Windows PowerShell style example (adjust as needed):

```powershell
scp -i .\YourMachineAddress.pem .\random_forest_model.pkl .\ohe.pkl ubuntu@YourMachineAddress.nus.cloud:~/helixhomies/
```

Linux/macOS style example (adjust as needed):

```bash
scp -i ./YourMachineAddress.pem ./random_forest_model.pkl ./ohe.pkl ubuntu@YourMachineAddress.nus.cloud:~/helixhomies/
```

3. Reconnect to your instance once the files are uploaded.

---

## 4. System updates & required Python packages

Run these commands on the Ubuntu instance:

```bash
sudo apt update
sudo apt install -y python3-pip
pip3 install numpy pandas matplotlib scikit-learn imbalanced-learn xgboost joblib orjson
```

---

## 5. Create the `helixhomies.py` script

Inside the `helixhomies` folder, create the Python script:

```bash
nano helixhomies.py
```

Copy and paste the following content into `helixhomies.py` (then press `Ctrl+O`, `Enter`, and `Ctrl+X` to save and exit):

```python
import gzip, json, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve
)

def extract_features(reads_list):
    """Extract statistical features from reads"""

    reads_array = np.array(reads_list)  # shape = (n_reads, 9)

    stats = []
    weight_map = {1: 0, 2: 0, 4: 3, 5: 3, 7: 6, 8: 6}  # Map feature indices for sd/mean to their dwell time col for weighting

    for i in range(reads_array.shape[1]):  # for each of the 9 features
        col = reads_array[:, i]
        if i in weight_map:
            w = reads_array[:, weight_map[i]] # corresponding dwell time column
            denom = np.sum(w) # sum of dwell times
            if denom > 0:
                mean_val = np.sum(w * col) / denom # weighted mean
                sd_val = np.sqrt(np.sum(w * (col - mean_val)**2) / denom) # weighted std
                stats.extend([mean_val, sd_val])

        stats.extend([
            np.mean(col),
            np.std(col),
            np.median(col),
            np.min(col),
            np.max(col)
        ])

    # compute diffs between prev-central and next-central
    diffs = {
        "prev": reads_array[:, 3:6] - reads_array[:, 0:3],   # central - previous
        "next": reads_array[:, 6:9] - reads_array[:, 3:6]    # next - central
    }

    # compute average dwell times for weighting
    dwells = {
        "prev": (reads_array[:, 3] + reads_array[:, 0]) / 2,  # avg dwell time between central and previous
        "next": (reads_array[:, 3] + reads_array[:, 6]) / 2    # avg dwell time between central and next
    }

    for key in ["prev", "next"]:
        diff = diffs[key]
        weights = dwells[key]
        denom = np.sum(weights)

        stats.extend([
            np.mean(diff, axis=0).tolist(),  # average per feature type
            np.std(diff, axis=0).tolist(),
            np.median(diff, axis=0).tolist(),
            np.min(diff, axis=0).tolist(),
            np.max(diff, axis=0).tolist()
        ])

        if denom > 0:
            for j in (1,2): # sd diff and mean diff
                w_mean = np.sum(weights * diff[:, j]) / denom
                w_std = np.sqrt(np.sum(weights * (diff[:, j] - w_mean)**2) / denom)
                stats.extend([w_mean, w_std]) # add in weighted mean and std for sd diff and mean diff

    # Flatten lists inside stats
    stats = np.concatenate([np.ravel(s) if isinstance(s, (list, np.ndarray)) else [s] for s in stats])

    return np.array(stats)


def extract_seq_features(df):
    """Extract sequence features from a sequence string."""
    pos = [0,1,2,5,6] # 3 and 4 are A and C from DRACH motif, will not change
    for i in pos:
        df[f'seq_{i}'] = df['sequence'].str[i]
    return df


def test_feature_extraction(test_df, encoder):
    """Carry out all feature extraction steps on the test data."""
    test_df = test_df.copy()
    test_df['features'] = test_df['reads'].apply(extract_features)
    n_features = len(test_df['features'].iloc[0])
    feature_columns = [f'feature_{i}' for i in range(n_features)]

    # Convert features list to separate columns
    features_df = pd.DataFrame(test_df['features'].tolist(), columns=feature_columns, index=test_df.index)

    # Extract sequence features
    ohe_columns = extract_seq_features(test_df)
    ohe_columns = encoder.transform(ohe_columns[['seq_0', 'seq_1', 'seq_2', 'seq_5', 'seq_6']])
    ohe_df = pd.DataFrame(ohe_columns, columns=encoder.get_feature_names_out(), index=test_df.index)

    # Combine all features
    test_df = pd.concat([features_df, ohe_df], axis=1)

    return test_df

import orjson
def parse_json_unzipped(path):
    json_data = []
    with open(path, 'r') as f:
        for line in f:
            entry = orjson.loads(line)
            for transcript, position_dict in entry.items():
                for position, sequence_dict in position_dict.items():
                    for sequence, reads in sequence_dict.items():
                        json_data.append({
                            'transcript_id': transcript,
                            'transcript_position': int(position),
                            'sequence': sequence,
                            'reads': reads
                        })

    df = pd.DataFrame(json_data)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'orig_idx'}, inplace=True)
    df.set_index('orig_idx', inplace=True)

    return df

import joblib

rf_model = joblib.load('/home/ubuntu/helixhomies/random_forest_model.pkl')
enc = joblib.load('/home/ubuntu/helixhomies/ohe.pkl')


# change the filename and path
filename = 'predictions'
filepath = '/home/ubuntu/helixhomies/data.json'  # assuming you've moved the JSON file here
data = parse_json_unzipped(filepath)
test = test_feature_extraction(data, enc)
pred = data.copy()
pred['score'] = rf_model.predict_proba(test)[:, 1]
final_pred = pred[['transcript_id', 'transcript_position', 'score']].copy()
final_pred.to_csv(f'/home/ubuntu/helixhomies/{filename}.csv', index=False)
```

---

## 6. Run the prediction script

From inside `/home/ubuntu/helixhomies` (or the helixhomies folder in your home directory), execute:

```bash
python3 helixhomies.py
```

---

## 7. If it crashes

If the job crashes due to memory/GPU constraints, try a machine with higher GPU RAM (or higher instance spec). This is unlikely but flagged just in case.

---

## 8. Verify output

When the script finishes, ensure there is a file called `predictions.csv` in the `helixhomies` folder:

```bash
ls
# you should see: predictions.csv
```

---

## 9. Download the `predictions.csv` to your local machine

1. Exit the instance:

```bash
exit
```

2. From your local machine run (edit `YourMachineAddress` accordingly):

Windows PowerShell style example (adjust as needed):

```powershell
scp -i YourMachineAddress.pem ubuntu@YourMachineAddress.nus.cloud:/home/ubuntu/helixhomies/predictions.csv .
```

Linux/macOS style example (adjust as needed):

```bash
scp -i ~/YourMachineAddress.pem ubuntu@YourMachineAddress.nus.cloud:/home/ubuntu/helixhomies/predictions.csv .
```

The file will appear in the same folder as your `.pem` file on your local machine.

---

## Notes / Tips

- Edit any URL, `YourMachineAddress`, and file paths to match your specific instance and file locations.
- If using Windows, pay attention to backslash vs forward slash in `scp` commands and use PowerShell or WSL accordingly.
- The code expects `data.json` to be in `/home/ubuntu/helixhomies/` and the two `.pkl` files to be present there as well.

---

## License / Acknowledgements

Provided for instructional use within the course. Cite the project and maintainers when sharing results.

