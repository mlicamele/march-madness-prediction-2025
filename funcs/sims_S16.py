import pandas as pd
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")

from funcs.ml_S16 import tourney_sim_test, get_teams, prep_model, train_model, get_teams, section

def simulate_bracket(YEAR: int, input_dir: str):

    X_train, y_train, X_test, y_test, team_key, test_case = prep_model(YEAR, input_dir)

    teams = get_teams(test_case)

    model = train_model(X_train, y_train)

    data = X_test.loc[teams]

    teams, rounds = tourney_sim_test(test_case, data, model, team_key)

    return teams, rounds

def simulate_probs(input_dir, output_dir, YEAR, sims, n):
    
    with open(f"test_cases/test_case_S16_{YEAR}.pkl", "rb") as f:
        test_case = pkl.load(f)

    avg_prob_df = None
    total_prob_df = None
    avg_df = None

    for i in range(1, sims + 1):

        teams = get_teams(test_case)

        X_train, y_train, X_test, y_test, team_key, test_case = prep_model(YEAR, input_dir)

        model = train_model(X_train, y_train)

        data = X_test.loc[teams]

        prob = model.predict_proba(data)

        _ = tourney_sim_test(test_case, data, model, team_key, False)

        prob_df = pd.DataFrame(data = prob, index = teams, columns = ["Champions", "Finals", "F4", "E8", "S16"])

        if total_prob_df is None:
            total_prob_df = prob_df
        else:
            total_prob_df = total_prob_df + prob_df
        
        avg_prob_df = total_prob_df / i
        avg_prob_df = avg_prob_df.round(3)

        avg_prob_df = avg_prob_df.iloc[:, :-1]

        if i % n == 0:
            print(f"Simulations: {i}")

    avg_prob_df = avg_prob_df.cumsum(axis=1)

    avg_prob_df = avg_prob_df.round(3)

    for col, i in zip(avg_prob_df.columns, range(len(avg_prob_df) - 1)):

            secs = section(test_case, i)

            for sec in secs:

                teams = get_teams(sec)

                total_prob_sec = 0

                for team_id in teams:

                    total_prob_sec += avg_prob_df.loc[team_id, col]
                
                for team_id in teams:
                    
                    avg_prob_df.loc[team_id, col] /= total_prob_sec

    avg_prob_df = avg_prob_df.iloc[:, ::-1]

    avg_prob_df = avg_prob_df.round(3)

    avg_df = pd.concat([team_key, avg_prob_df], join = "inner", axis = 1)

    capital_teams = avg_df["TEAM"]

    avg_df["TEAM"] = [str(v).lower() for v in avg_df["TEAM"]]

    avg_df = avg_df.sort_values(by = "TEAM", ascending=True)

    avg_df["TEAM"] = capital_teams

    avg_df.to_csv(f"{output_dir}/avg_prob_S16_df.csv")

    return avg_df