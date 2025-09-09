from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import pickle as pkl
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

def get_teams(list):

    if type(list[0]) == int:
        result = list
    else:
        list0 = get_teams(list[0])
        list1 = get_teams(list[1])

        result = []

        for i in list0:
            result.append(i)
        for i in list1:
            result.append(i)
        
    return result

def analysis(y_test: list, preds: list):

    labels = [16, 8, 4, 2, 1]

    cm = confusion_matrix(y_test, preds, labels = labels)

    print("Confusion Matrix:")

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot()

    plt.show()

    print()
    print(f"Bracket Score: {bracket_score(y_test, preds)}")
    print()
    print(f"R Score: {r2_score(y_test, preds)}")
    print()
    print(f"Accuracy Score: {accuracy_score(y_test, preds)}")
    print()

def bracket_score(y_true, y_pred):

    score = 0

    round = [1, 2, 4, 8, 16, 32, 64]
    reverse_rounds = round.copy()
    reverse_rounds.reverse()
    round_scores = [0, 0, 0, 0, 0, 0, 0]

    for y, pred in zip(y_true, y_pred):

        if y >= pred:
            i = round.index(y)
        else:
            i = round.index(pred)

        cur = reverse_rounds[i] / 2

        while cur >= 1:
            round_scores[i] += int(cur)
            i += 1
            cur /= 2
    
    round_scores = round_scores[:-3]

    score = np.sum(round_scores)
    
    return score, round_scores

def estimate_upsets(teams: list, rounds: list, names: pd.DataFrame):

    count = 0

    for team, round in zip(teams, rounds):
        
        seed = names.loc[team]["SEED"]

        if round == 32 and seed >= 11:
            count += 1
        elif round == 16 and seed >= 7:
            count += 1
        elif round == 8 and seed >= 5:
            count += 1
        elif round == 4 and seed >= 3:
            count += 1
    
    return count

def prep_model(year: int, input_directory: str):

    with open(os.path.join("test_cases", f"test_case_S16_{year}.pkl"), "rb") as f:

        test_case: list = pkl.load(f)

    team_key = pd.read_csv(os.path.join(input_directory, "team_key.csv"), index_col=0)

    teams = get_teams(test_case)

    train = pd.read_csv(os.path.join(input_directory, f"train_{year}.csv"), index_col = 0)
    train = train.loc[train["ROUND"] < 32]

    test = pd.read_csv(os.path.join(input_directory, f"test_{year}.csv"), index_col = 0)
    test = test.loc[teams]

    X_train = train.drop(columns = ["ROUND"])
    y_train = train["ROUND"]

    X_test = test.drop(columns = ["ROUND"])
    y_test = test["ROUND"]

    scaler = StandardScaler()

    X = pd.concat([X_train, X_test])

    cols = X.columns

    scaler.fit(X)

    X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)

    X_test = pd.DataFrame(X_test, index=teams,columns=cols)

    return X_train, y_train, X_test, y_test, team_key, test_case

def train_model(X_train, y_train):
    
    model = RandomForestClassifier().fit(X_train, y_train)

    return model

def predict_round(bracket_section: list, advanced_teams: list, data: pd.DataFrame, n: int, model: RandomForestClassifier):

    # weights: np.ndarray = np.array([6, 5, 4, 3, 2, 1])

    teams = get_teams(bracket_section)

    data = data.loc[teams]

    for team in advanced_teams:
        if team in teams:
            return -1
    
    prob = model.predict_proba(data)

    # prob = np.average(prob[:, :n + 1], weights = weights[:n+1], axis = 1)
    # prob = np.average(prob[:, :n + 1], axis = 1)
    prob = prob[:, n]

    i = prob.argmax()

    id = data.iloc[i].name
    
    return id

def predict_round_baseline(bracket_section: list, advanced_teams: list, data: pd.DataFrame, team_key: pd.DataFrame):

    teams = get_teams(bracket_section)

    data = data.loc[teams]

    team_key = team_key.loc[teams]

    for team in teams:
        if team in advanced_teams:
            return -1
    
    i = team_key["SEED"].argmin()

    id = team_key.iloc[i].name

    return id

def predict_round_perfect(bracket_section: list, advanced_teams: list, data: pd.DataFrame, team_key: pd.DataFrame):

    teams = get_teams(bracket_section)

    data = data.loc[teams]

    team_key = team_key.loc[teams]

    for team in teams:
        if team in advanced_teams:
            return -1
    
    i = team_key["ROUND"].argmin()

    id = team_key.iloc[i].name

    return id

def section(bracket: list, s: int):
    
    result = []

    return sec_helper(bracket, s, result)

def sec_helper(bracket: list, s: int, result: list):
    
    if s > 0:
        result = sec_helper(bracket[0], s - 1, result)
        result = sec_helper(bracket[1], s - 1, result)
    else:
        result.append(bracket)
    
    return result

def tourney_sim_perfect(bracket: list, data: pd.DataFrame, names: pd.DataFrame, prnt: bool = True):

    teams = get_teams(bracket)

    names = names.loc[teams]

    advanced_teams = []
    rounds = []
    NAMES = ["Champion", "Finals", "Final Four", "Elite Eight"]
    ROUND_OPTIONS = [1, 2, 4, 8]

    for n in range(len(NAMES)):
        
        if prnt:

            print(NAMES[n])
        
        secs = section(bracket, n)

        for sec in secs:

            i = predict_round_perfect(sec, advanced_teams, data, names)

            if i == -1:

                continue
            
            if prnt:

                print(names.loc[i])
                print()

            advanced_teams.append(i)
            rounds.append(ROUND_OPTIONS[n])
        
        if prnt:

            print()
    
    for team in teams:

        if team not in advanced_teams:

            advanced_teams.append(team)
            rounds.append(16)

    return advanced_teams, rounds

def tourney_sim_baseline(bracket: list, data: pd.DataFrame, names: pd.DataFrame, prnt: bool = True):

    teams = get_teams(bracket)

    names = names.loc[teams]

    advanced_teams = []
    rounds = []
    NAMES = ["Champion", "Finals", "Final Four", "Elite Eight"]
    ROUND_OPTIONS = [1, 2, 4, 8]

    for n in range(len(NAMES)):
        
        if prnt:

            print(NAMES[n])
        
        secs = section(bracket, n)

        for sec in secs:

            i = predict_round_baseline(sec, advanced_teams, data, names)

            if i == -1:

                continue
            
            if prnt:

                print(names.loc[i])
                print()

            advanced_teams.append(i)
            rounds.append(ROUND_OPTIONS[n])
        
        if prnt:

            print()
    
    for team in teams:

        if team not in advanced_teams:

            advanced_teams.append(team)
            rounds.append(16)

    return advanced_teams, rounds

def tourney_sim_test(bracket: list, data: pd.DataFrame, model: RandomForestClassifier, names: pd.DataFrame, prnt: bool = True):

    teams = get_teams(bracket)

    names = names.loc[teams]

    advanced_teams = []
    rounds = []
    NAMES = ["Champion", "Finals", "Final Four", "Elite Eight"]
    ROUND_OPTIONS = [1, 2, 4, 8]

    for n in range(len(NAMES)):
        
        if prnt:

            print(NAMES[n])
        
        secs = section(bracket, n)

        for sec in secs:

            i = predict_round(sec, advanced_teams, data, n, model)

            if i == -1:

                continue
            
            if prnt:

                print(names.loc[i])
                print()

            advanced_teams.append(i)
            rounds.append(ROUND_OPTIONS[n])
        
        if prnt:

            print()
    
    for team in teams:

        if team not in advanced_teams:

            advanced_teams.append(team)
            rounds.append(16)

    return advanced_teams, rounds

def simulate_brackets(input_directory: str, year: int, n: int, n_mod: int):

    total_score = 0
    avg_score = 0
    max_score = 0
    total_round_scores = [0, 0, 0, 0]
    avg_round_scores = [0, 0, 0, 0]
    max_round_scores = [0, 0, 0, 0]

    total_upsets = 0
    avg_upsets = 0

    best_model = None

    X_train, y_train, X_test, y_test, team_key, test_case = prep_model(year, input_directory)

    for i in range(1, n + 1):
        
        model = train_model(X_train, y_train)
        
        teams, preds = tourney_sim_test(test_case, X_test, model, team_key, False)
        y_test: pd.Series = y_test.loc[teams]

        score, round_scores = bracket_score(y_test, preds)

        upsets = estimate_upsets(teams, preds, team_key)

        total_upsets += upsets
        avg_upsets = total_upsets / i

        total_score += score
        avg_score = total_score / i

        for j in range(len(total_round_scores)):
            total_round_scores[j] += round_scores[j]
            avg_round_scores[j] = int(total_round_scores[j] / i)
        
        if score > max_score:
            max_score = score
            max_round_scores = round_scores
            best_model = model

        if i % n_mod == 0:
            print(f"Brackets Simulated:\t\t{i}")
            print(f"Average Upsets:\t\t\t{avg_upsets}")
            print(f"Average Bracket Score:\t\t{avg_score}")
            print(f"Average Bracket Round Scores:\t{avg_round_scores}")
            print(f"Max Bracket Score:\t\t{max_score}")
            print(f"Max Bracket Round Scores:\t{max_round_scores}")
            print()
        
    return best_model, X_train, y_train, X_test, y_test, team_key, test_case, avg_score, max_score