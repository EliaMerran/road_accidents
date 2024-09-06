# Road Accidents Analysis and Prediction

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.8 or higher

### Installation

1. Clone the repository
```bash
git clone https://github.com/EliaMerran/road_accidents.git
```

### Folders and Files
- `data/`: contains the dataset
- `configurations/`: contains configuration files of the models
- `notebooks/`: contains the Jupyter notebooks
- `models/`: contains the trained models with their configuration
- `model_data_setup.py`: setup accidents and cluster data from raw data.
- `model_training.py`: train the models.
- `model_performence_overview.py`: compare and evaluate the model's performance.
- `preprocess.py`: preprocess the data, creates the clusters and more.
- `graphs_templates.py`: contains templates for useful graphs.
- `utilities.py`: contains utility functions.
- `theoretical_overview.py`: create all theoretical overview tables. 

### Setup
#### Option 1:
1. Copy Israel.zip and United Kingdom.zip to data/ folder and extract them.
#### Option 2:
1. Copy Israel data under data/raw_data
2. Download United Kingdom data from https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data
3. Move '[dft-road-casualty-statistics-collision-1979-latest-published-year.csv](data%2Fraw%20data%2Fuk%2Fdft-road-casualty-statistics-collision-1979-latest-published-year.csv)' to data/United Kingdom and rename it to 'Accidents.csv'
4. Run model_data_setup.py

#### Comment: for the theoretical_overview workflow you need the Israel raw data under data/raw data (Option 2).