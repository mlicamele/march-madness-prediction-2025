# march-madness-prediction-2025

A comprehensive machine learning project for predicting NCAA Men's Basketball Tournament (March Madness) outcomes using advanced statistical modeling and bracket simulation.

## Project Overview

This project uses machine learning algorithms to predict March Madness tournament results by analyzing historical basketball data, team statistics, and performance metrics. The system can simulate entire tournament brackets and evaluate model performance against actual tournament outcomes.

### Key Features

- **Tournament Simulation**: Full bracket simulation
- **Sweet 16 Analysis**: Specialized modeling for teams that advance to later rounds
- **Performance Analysis**: Comprehensive evaluation using bracket scoring and statistical metrics
- **Feature Engineering**: Advanced basketball statistics and data processing pipeline

## Project Structure

```
march_madness/
├── data.ipynb                     # Data processing and feature creation
├── dataset_test.ipynb             # Model testing and validation
├── sims.ipynb                     # Tournament probability simulations
├── sims_S16.ipynb                 # Sweet 16 specific simulations
├── simulate_bracket.ipynb         # Full bracket simulation
├── simulate_bracket_S16.ipynb     # Sweet 16 bracket simulation
├── test_cases.ipynb               # Test case creation for validation
├── testing.ipynb                  # Model testing and evaluation
├── testing_S16.ipynb              # Sweet 16 model testing
├── funcs/                         # Core functionality modules
│   ├── data.py                    # Data processing functions
│   ├── ml.py                      # Machine learning models
│   ├── ml_S16.py                  # Sweet 16 specific models
│   ├── sims.py                    # Simulation functions
│   └── sims_S16.py                # Sweet 16 simulations
├── data/                          # Original datasets
├── processed_data/                # Processed datasets
├── test_cases/                    # Validation test cases
└── output/                        # Model outputs and results
```

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/march_madness.git
cd march_madness
```

2. **Install dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib
```

3. **Set up environment**
```bash
# Ensure Python 3.12+ is installed
python --version
```

## Quick Start

### 1. Data Processing
```python
# Process historical tournament data
from funcs.data import process_years
process_years("data", "processed_data", 2021, 2023, 
              AP6=True, bart_a=True, # ... feature flags)
```

### 2. Run Tournament Simulation
```python
from funcs.ml import simulate_brackets
model, X_train, y_train, X_test, y_test, team_key, test_case, avg_score, max_score = 
    simulate_brackets(input_directory="processed_data", year=2025, n=25)
```

### 3. Generate Predictions
```python
from funcs.sims import simulate_probs
df = simulate_probs("processed_data", "output", 2025, sims=1000, n=100)
```

## Model Features

The project includes several sophisticated features for basketball analytics:

### Statistical Features
- **AP6**: Associated Press Top 25 rankings
- **BART**: Barthag power rating
- **Conference Strength**: Conference performance metrics
- **KenPom Ratings**: Advanced efficiency metrics
- **Seed Performance**: Historical seed-based analysis
- **Team Rankings**: Multiple ranking system integrations

### Evaluation Metrics
- **Bracket Scoring**: Traditional tournament scoring (10-20-40-80-160-320 points)
- **Accuracy**: Game prediction accuracy
- **Confusion Matrix**: Classification performance visualization

## Results & Performance

The system provides comprehensive analysis including:
- Average bracket scores across multiple simulations
- Upset prediction capabilities
- Round-by-round performance analysis
- Champion and Final Four probability distributions

### Sample Output
```
Brackets Simulated: 25
Average Upsets: 2.52
Average Bracket Score: 126.44
Average Bracket Round Scores: [29, 26, 14, 12, 23, 20]
Max Bracket Score: 155
```

## 🔬 Advanced Features

### Tournament Bracket Structure
The system models the complete 64-team tournament structure:
- **Round of 64**: First round 
- **Round of 32**: Second round
- **Sweet 16**: Regional semifinals
- **Elite 8**: Regional finals
- **Final Four**: National semifinals
- **Championship**: National final

### Sweet 16 Specialization
Specialized modeling for teams that advance deep in the tournament, recognizing that different factors become important in later rounds.

## Usage Examples

### Create Tournament Predictions
```python
# Generate probability matrix for all possible matchups
YEAR = 2025
sims = 1000
df = simulate_probs("processed_data", "output", YEAR, sims, n=100)
print(df.sort_values(by=["SEED"], ascending=True))
```

### Simulate Single Bracket
```python
# Simulate one complete bracket
from funcs.sims import simulate_bracket
bracket_result = simulate_bracket(2025, "processed_data")
```

### Compare Model to Baseline
```python
# Compare different model approaches
teams, ml_preds = tourney_sim_test(test_case, X_test, model, team_key)
teams, seed_preds = tourney_sim_baseline(test_case, X_test, team_key)
teams, perfect_preds = tourney_sim_perfect(test_case, X_test, team_key)
```

## Testing & Validation

The project includes comprehensive testing infrastructure:
- Historical tournament validation (2021-2024)
- Cross-validation across multiple years
- Performance comparison against baseline methods

## Data Sources

The project processes various basketball statistics including:
- Game-by-game results
- Team efficiency metrics
- Player statistics and ratings
- Conference strength indicators
- Historical tournament performance

*"The beauty of March Madness lies in its unpredictability, but with data and machine learning, we can better understand the madness."*
