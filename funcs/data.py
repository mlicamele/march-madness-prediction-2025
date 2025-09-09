import pandas as pd
import os

def process_years(input_directory: str, output_directory: str, start_year: int, end_year: int, AP6: bool = True, bart_a: bool = True, bart_h: bool = True, bart_n: bool = True, conf: bool = True, evan: bool = True, hc_rate: bool = True, hc_index: bool = True, ken_pre: bool = True, pre: bool = True, resume: bool = True, seed: bool = True, shooting: bool = True, team_rank: bool = True, s: bool = True, rank: bool = True):

    for year in range(start_year, end_year + 1):

        process_data(input_directory, output_directory, year, AP6, bart_a, bart_h, bart_n, conf, evan, hc_rate, hc_index, ken_pre, pre, resume, seed, shooting, team_rank, s, rank)

def process_data(input_directory: str, output_directory: str, training_year: int, AP6: bool = True, bart_a: bool = True, bart_h: bool = True, bart_n: bool = True, conf: bool = True, evan: bool = True, hc_rate: bool = True, hc_index: bool = True, ken_pre: bool = True, pre: bool = True, resume: bool = True, seed: bool = True, shooting: bool = True, team_rank: bool = True, s: bool = True, rank: bool = True):

    df = None
    team_index = None

    files = sorted(os.listdir(input_directory))
    temp = files[0]
    files[0] = files[1]
    files[1] = temp

    for file in files:

        if file.startswith("Bart"):

            drop = ["TEAM ID", "ROUND", "GAMES", "W", "L", "SEED"]

            if "Neutral" in file and "Away" in file:
                h = "A-N"
            else:
                h = file.split(" ")[1][0]
                drop.append("TEAM")
                drop.append("YEAR")

            if not bart_a and h == "A":

                continue
                
            if not bart_h and h == "H":

                continue
            
            if not bart_n and h == "N":

                continue

            new_df = pd.read_csv(os.path.join(input_directory, file), index_col = 1)
            
            new_df = new_df.loc[new_df["YEAR"] >= 2013]

            new_df = new_df.drop(columns = drop)

            new_df.columns = [f"{c}_{h}" if c != "TEAM" else "TEAM" for c in new_df.columns]

            new_df.columns = [c if c != "YEAR_A-N" else "YEAR" for c in new_df.columns]

            if h == "A-N":
                team_index = new_df.index
            else:
                new_df = new_df.loc[team_index]

        elif "KenPom Bart" in file:

            drop = ["YEAR", "CONF", "QUAD NO", "QUAD ID", "TEAM ID", "TEAM"]

            new_df = pd.read_csv(os.path.join(input_directory, file), index_col = 5)

            new_df = new_df.drop(columns = drop)

            new_df = new_df.loc[team_index]
        
        elif "Evan" in file and evan:

            drop = ["YEAR", "TEAM", "SEED", "ROUND"]

            new_df = pd.read_csv(os.path.join(input_directory, file), index_col = 1)

            for c in new_df.columns:
                if "KILL" in c:
                    drop.append(c)

            new_df = new_df.drop(columns = drop)

            new_df = new_df.loc[team_index]
        
        elif "Index" in file and hc_index:

            drop = ["YEAR", "TEAM", "SEED", "ROUND", "WINS", "DRAW"]

            new_df = pd.read_csv(os.path.join(input_directory, file), index_col = 1)

            new_df = new_df.drop(columns = drop)

            new_df = new_df.loc[[i for i in new_df.index if i in team_index]]
        
        elif "KenPom Pre" in file and ken_pre:

            drop = ["YEAR", "TEAM", "SEED", "ROUND"]

            new_df = pd.read_csv(os.path.join(input_directory, file), index_col = 1)

            new_df = new_df.loc[new_df["SEED"] >= 1]

            new_df = new_df.drop(columns = drop)

            new_df = new_df.loc[team_index]

        elif "Resumes" in file and resume:

            drop = ["YEAR", "TEAM", "SEED", "ROUND"]

            new_df = pd.read_csv(os.path.join(input_directory, file), index_col = 1)

            new_df = new_df.drop(columns = drop)

            new_df["BID TYPE"] = [0 if b == "Auto" else 1 for b in new_df["BID TYPE"]]

            new_df = new_df.loc[team_index]
        
        elif "Shooting" in file and shooting:

            drop = ["YEAR", "TEAM", "TEAM ID"]

            new_df = pd.read_csv(os.path.join(input_directory, file), index_col = 1)

            new_df = new_df.drop(columns = drop)

            new_df = new_df.loc[team_index]
        
        elif "Preseason Votes" in file and pre:

            drop = ["YEAR", "TEAM", "SEED", "ROUND"]

            new_df = pd.read_csv(os.path.join(input_directory, file))

            new_df = new_df.loc[new_df["TEAM NO"] >= 1]

            new_df = new_df.set_index(keys = ["TEAM NO"], drop = True)

            new_df = new_df.drop(columns = drop)

            new_df.columns = [i + "_PRE" for i in new_df.columns]

            new_df = new_df.loc[[i for i in new_df.index if i in team_index]]

        elif "Heat Check Ratings" in file and hc_rate:

            drop = ["YEAR", "TEAM", "SEED", "ROUND"]

            new_df = pd.read_csv(os.path.join(input_directory, file), index_col = 1)

            new_df = new_df.drop(columns = drop)

            for col in new_df.columns:

                new_df[col] = new_df[col].astype(int)

            new_df = new_df.loc[[i for i in new_df.index if i in team_index]]
        
        elif "AP" in file and AP6:

            drop = ["YEAR", "TEAM", "SEED", "ROUND"]

            new_df = pd.read_csv(os.path.join(input_directory, file))

            new_df = new_df.loc[new_df["TEAM NO"] >= 1]
            
            new_df["TEAM NO"] = new_df["TEAM NO"].astype(int)

            new_df = new_df.set_index(keys = ["TEAM NO"], drop = True)

            new_df = new_df.drop(columns = drop)

            new_df.columns = [i + "_AP6" for i in new_df.columns]

            new_df = new_df.loc[[i for i in new_df.index if i in team_index]]
        
        elif "TeamRank" in file and team_rank:

            drop = ["YEAR", "TEAM", "SEED", "ROUND"]

            new_df = pd.read_csv(os.path.join(input_directory, file), index_col = 1)

            new_df = new_df.drop(columns = drop)

            new_df = new_df.loc[team_index]
        
        else:
            continue
        
        if df is None:

            df = new_df

        else:

            df = pd.concat([df, new_df], axis = 1)

    if conf:

        new_df = pd.read_csv(os.path.join(input_directory, "Conference Stats.csv"))

        result = []

        for index in df.index:

            row = df.loc[index]
            
            year = row["YEAR"]
            conf = row["CONF"]

            conf_row: pd.DataFrame = new_df.loc[(new_df["YEAR"] == year) & (new_df["CONF"] == conf)]

            if conf_row.empty:
                conf_row = pd.DataFrame(new_df.iloc[0], index = [index], columns = new_df.columns)
                conf_row[conf_row.columns] = 0
            else:
                conf_row = pd.DataFrame(conf_row)
                conf_row.index = [index]

            result.append(conf_row)

        conf_df = pd.concat(result)

        conf_df.columns = [f"{c}_conf" for c in conf_df.columns]

        df = pd.concat([df, conf_df], axis = 1)
    
    if seed:

        new_df = pd.read_csv(os.path.join(input_directory, "Seed Results.csv"))

        result = []

        for index in df.index:

            row = df.loc[index]
            
            seed = row["SEED"]

            seed_row = new_df.loc[(new_df["SEED"] == seed)]
            seed_row = pd.DataFrame(seed_row)
            seed_row.index = [index]

            result.append(seed_row)

        seed_df = pd.concat(result)

        seed_df.columns = [f"{c}_seed" for c in seed_df.columns]

        df = pd.concat([df, seed_df], axis = 1)
    
    # All
    if seed and conf:

        drop = ["CONF", "CONF ID", "CONF ID_conf", "CONF_conf", "YEAR_conf", "SEED_seed", "GAMES", "GAMES_seed", "L", "L_conf", "L_seed", "W", "W_conf", "W_seed", "TOP2_seed"]

    # No seed only
    elif conf:
        
        drop = [i for i in drop if "_seed" in i]

    # No conf only
    elif seed:

        drop = [i for i in drop if "_conf" in i]

    # No conf or seed
    else:

        drop = [i for i in drop if ("_conf" in i or "_seed" in i)]

    df = df.drop(columns = drop)

    if rank:

        drop = [c for c in df.columns if "RANK" in c]

        df = df.drop(columns = drop)

    df = df.replace('%', '', regex=True)

    df = df.fillna(0)

    df = df.loc[df["ROUND"] != 68]

    df.index = df.index.astype(int)

    if not os.path.exists(output_directory):

        os.mkdir(output_directory)

    team_key = df[["YEAR", "TEAM", "SEED", "ROUND"]]
    team_key.to_csv(os.path.join(output_directory, "team_key.csv"))

    drop = ["YEAR", "TEAM"]
    
    if s:

        drop.append("SEED")

    train_df = df.loc[df["YEAR"] != training_year]
    test_df = df.loc[df["YEAR"] == training_year]

    df = df.drop(columns = drop)
    train_df = train_df.drop(columns = drop)
    test_df = test_df.drop(columns = drop)

    df.to_csv(os.path.join(output_directory, "mm_data.csv"))

    train_df.to_csv(os.path.join(output_directory, f"train_{training_year}.csv"))
    test_df.to_csv(os.path.join(output_directory, f"test_{training_year}.csv"))