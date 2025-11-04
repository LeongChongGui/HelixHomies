#!/usr/bin/env python3

import sys
import gzip
import orjson
import joblib
import numpy as np
import pandas as pd

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

def parse_json(path):
    json_data = []
    if path.endswith('.gz'):
        with gzip.open(path, 'rt') as f:
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
    elif path.endswith('.json'):
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

def main():
    if len(sys.argv) < 2:
        print('Usage: python predict_m6a.py <input_data_path> <output_path>')
        sys.exit(1)
    else:
        print(f'Input data path: {sys.argv[1]}')
        print(f'Output path: {sys.argv[2]}')

    # Load encoder and trained model
    print('Loading model and encoder...')
    rf_model = joblib.load('./random_forest_model.pkl')
    enc = joblib.load('./ohe.pkl')
    print('Model and encoder loaded.')

    # Predicting m6A modifications on dataset3 

    # Load data
    print('Loading data...')
    DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else './dataset3.json.gz'
    OUTPUT_PATH = sys.argv[2] if len(sys.argv) > 2 else './predictions.csv'
    data = parse_json(DATA_PATH)
    print('Data loaded.')

    # Extract features
    print('Extracting features...')
    test = test_feature_extraction(data, enc)
    print('Features extracted.')

    # Predict
    print('Making predictions...')
    pred = data[['transcript_id', 'transcript_position']].copy()
    pred['score'] = rf_model.predict_proba(test)[:, 1]

    # Save predictions
    pred.to_csv(OUTPUT_PATH, index=False)
    print(f'Predictions saved to {OUTPUT_PATH}.')

if __name__ == "__main__":
    main()