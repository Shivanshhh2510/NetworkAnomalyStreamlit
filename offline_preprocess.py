import pandas as pd
import joblib

# 1) column names
cols = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate",
    "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"
]

# 2) load raw KDD data (place 'kddcup.data_10_percent' in this folder)
df = pd.read_csv("kddcup.data_10_percent", names=cols)

# 3) create binary attack label: 0 = normal, 1 = attack
df["attack_type"] = (df["label"] != "normal.").astype(int)

# 4) drop unused columns
df = df.drop(columns=["label","num_outbound_cmds"])

# 5) one-hot encode categorical features
df = pd.get_dummies(df, columns=["protocol_type","service","flag"])

# 6) load your fitted scaler
scaler = joblib.load("scaler.pkl")
X = df.drop(columns=["attack_type"])
X_scaled = scaler.transform(X)

# 7) rebuild DataFrame & save to CSV
pre = pd.DataFrame(X_scaled, columns=X.columns)
pre["attack_type"] = df["attack_type"].values
pre.to_csv("preprocessed.csv", index=False)

print("Wrote preprocessed.csv with shape", pre.shape)
