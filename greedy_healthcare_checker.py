import pandas as pd, os

df_save_path = "./preprocessing/saved_dfs"
nia_list = [2, 3, 6, 21, 22, 24, 25, 154, 155, 27, 28, 43, 44,
            55, 61, 72, 85, 91, 110, 111, 112, 113, 115, 121,
            124, 125, 135, 136, 141, 142, 138, 139]

for nia in nia_list:
    fp = os.path.join(df_save_path, f"NIA_{nia}_healthcare.pkl")
    
    if os.path.exists(fp):
        df = pd.read_pickle(fp)
        count = len(df)

        print(f"NIA {nia}: {count} existing healthcare amenities")
        #print(f"NIA {nia}: {count} amenities | columns: {list(df.columns)}")

        if count == 0:
            print(f"NIA {nia}: NO existing healthcare → greedy scores slightly off")
    else:
        print(f"NIA {nia}: healthcare pkl missing")