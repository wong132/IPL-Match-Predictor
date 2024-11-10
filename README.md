# IPL-Match-Predictor

This project aims to predict the outcome of Indian Premier League (IPL) matches using historical data. The predictor utilizes various features such as wickets left, current run rate, required run rate, and remaining balls to estimate the likelihood of a team winning or losing.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Contributing](#contributing)
- [License](#license)

## Overview

The IPL Match Predictor processes a dataset of past IPL matches, applying transformations to calculate key metrics and form a data frame suitable for predictive modeling. The final dataset, `final_df`, contains the necessary features to train a model for match outcome prediction.

## Features

- Calculates wickets left based on player dismissals.
- Computes current run rate (`crr`) and required run rate (`rrr`) dynamically.
- Prepares a `result` column indicating match outcomes (win/loss).
- Assembles a final dataset ready for machine learning models.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/wong132/IPL-Match-Predictor.git
   cd IPL-Match-Predictor
   ```


## Usage

1. Load your IPL dataset into the project as `delivery_df`.
2. Run the code to process the data, calculate metrics, and prepare the final data frame (`final_df`).
3. Use `final_df` as input for machine learning models to predict match outcomes.

## Code Explanation

- **Wickets Calculation**: Calculates the remaining wickets for each match based on player dismissals.
    ```python
    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: x if x == "0" else "1")
    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')
    wickets = delivery_df.groupby('match_id').cumsum()['player_dismissed'].values
    delivery_df['wickets'] = 10 - wickets
    ```

- **Current Run Rate (`crr`)**: Calculates `crr` as runs scored per over based on current score and balls left.
    ```python
    delivery_df['crr'] = (delivery_df['current_score'] * 6) / (120 - delivery_df['balls_left'])
    ```

- **Required Run Rate (`rrr`)**: Calculates `rrr` as required runs per over.
    ```python
    delivery_df['rrr'] = (delivery_df['runs_left'] * 6) / delivery_df['balls_left']
    ```

- **Result Column**: Assigns `1` if the batting team won, otherwise `0`.
    ```python
    delivery_df['result'] = delivery_df.apply(lambda row: 1 if row['batting_team'] == row['winner'] else 0, axis=1)
    ```

- **Final Data Preparation**: Creates `final_df` with selected columns.
    ```python
    final_df = delivery_df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr', 'result']]
    ```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any feature requests or bug reports.

## License

This project is licensed under the Apache 2.0 license. See the [LICENSE](LICENSE) file for more details.
```

This README should serve as a clear introduction to your project, code, and how others can use and contribute to it. Let me know if you want to add any specific details about your predictor model!