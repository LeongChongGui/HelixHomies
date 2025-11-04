# Predicting m6A modification probabilities

This README teaches users how to predict m6A modification probabilities on a dataset using the pre-trained Random Forest model and one-hot encoder provided. The dataset should be a JSON file (possibly zipped) containing the following information: transcript id, position, sequence, and reads.

Follow the steps in order.

---

## 1. Machine specification (Ubuntu 22.04)

Ensure your Ubuntu machine is set to the following specs:

- **T3.2XLarge (32Gb RAM, 8 vCPUs)**

  This can be chosen by clicking the **pencil** icon on your machine type ➜ **General Purpose Machine** ➜ scroll down to **T3** ➜ select **T3.2XLarge**.

- **50Gb SSD**

---

## 2. Folder & initial data download (assumes you're at your home directory after connecting to the instance)

Connect to the instance using the Lecture / week 7 instructions.

Once connected, run the following commands in sequence:

```bash
mkdir helixhomies
cd helixhomies
# the next command should only be executed if you would like to make predictions on the SG-NEx data
# edit the URL to access the specific json file
wget http://sg-nex-data.s3-website-ap-southeast1.amazonaws.com/data/processed_data/m6Anet/SGNex_MCF7_directRNA_replicate3_run1/data.json
```

Note: If you DO NOT download the SG-NEx data directly into the server directory with the `wget` command, you will have to transfer a JSON file in the next step.

---

## 3. Transfer data file, `predict_m6a.py`, `ohe.pkl`, and `random_forest_model.pkl` from GitHub/local machine

Download `predict_m6a.py`, `ohe.pkl`, and `random_forest_model.pkl` (from our GitHub) to the same folder as your `.pem` file on your local machine. The two `.pkl` files contain stored artifacts so no retraining is required on dataset0. The `.py` file is the script used to generate predictions.

If you do not have a dataset to make predictions on (already stored in your local machine) and you did not directly download the SG-NEx data into the server directory, you may also download any of the files from our GitHub's `data/` directory to the same folder as all the other files.

**Workflow (disconnect, SCP, reconnect):**

1. Exit the instance:

```bash
exit
```

2. From your local machine (where all `.pem`, `.pkl`, and `.py` files are), run the `scp` command. Edit `YourMachineAddress` to your machine's address and adjust path separators as appropriate for your OS.

If you did not download the SG-NEx data in Step 1, transfer your JSON file in this step as well. If you would like to make predictions on a few data files, transfer them all by adding the path to the files as shown in the example.

Windows PowerShell Environment (adjust as needed):

```powershell
scp -i ./YourMachineAddress.pem ./random_forest_model.pkl ./ohe.pkl ./predict_m6a.py ./data.json.gz ubuntu@YourMachineAddress.nus.cloud:~/helixhomies/
# ensure current directory is where all necessary files are stored
# example: transfer 2 data files
# scp -i ./dsa4262-2510-teamname-myname.pem ./random_forest_model.pkl ./ohe.pkl ./predict_m6a.py ./data.json ./data2.json.gz ubuntu@dsa4262-2510-teamname-myname.nus.cloud:~/helixhomies/
```

Linux/macOS Environment (adjust as needed):

```bash
chmod 600 ./YourMachineAddress
scp -i ./YourMachineAddress.pem ./random_forest_model.pkl ./ohe.pkl ./predict_m6a.py ./data.json.gz ubuntu@YourMachineAddress.nus.cloud:~/helixhomies/
```

3. Reconnect to your instance once the files are uploaded and `cd helixhomies` to navigate to the helixhomies folder.

Run the `ll` command in the folder to check if all necessary files have been transferred.

- `random_forest_model.pkl`

- `ohe.pkl`

- `predict_m6a.py`

- `data.json` (or `data.json.gz`, according to your file name)

---

## 4. System updates & required Python packages

Run these commands on the Ubuntu instance:

```bash
sudo apt update
sudo apt install -y python3-pip
pip3 install numpy pandas scikit-learn joblib orjson
```

---

## 5. Run the prediction script

From inside `/home/ubuntu/helixhomies` (or the helixhomies folder in your home directory), execute:

```bash
python3 predict_m6a.py input_data_path output_data_path
# e.g. python3 predict_m6a.py ./data.json ./predictions_csv
```

---

## 6. If it crashes

If the job crashes due to memory/GPU constraints, try a machine with higher GPU RAM (or higher instance spec). This is unlikely but flagged just in case.

---

## 7. Verify output

When the script finishes, ensure there is a file called `predictions.csv` (or the name you set the output file to be) in the `helixhomies` folder. To view the content of the file, use the `less` command.

```bash
ls
# you should see: predictions.csv
```

```bash
less predictions.csv
# you will be able to view the file
```

---

## 8. Download the `predictions.csv` to your local machine

1. Exit the instance:

```bash
exit
```

2. From your local machine run (edit `YourMachineAddress` accordingly):

```bash
scp -i YourMachineAddress.pem ubuntu@YourMachineAddress.nus.cloud:/home/ubuntu/helixhomies/predictions.csv .
# this saves it in your current directory
# edit the last argument . to ./path_to_desired_directory if needed
```

The file will appear in the same folder as your `.pem` file on your local machine.

---

## Notes / Tips

- Edit any URL, `YourMachineAddress`, and file paths to match your specific instance and file locations.
- If using Windows, pay attention to backslash vs forward slash in `scp` commands and use PowerShell or WSL accordingly.
- The code expects `data.json` to be in `/home/ubuntu/helixhomies/` and the two `.pkl` files and prediction script `predict_m6a.py` to be present there as well.

---

## License / Acknowledgements

Provided for instructional use within the course. Cite the project and maintainers when sharing results.

