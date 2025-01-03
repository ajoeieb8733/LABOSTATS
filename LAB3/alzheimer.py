"""Class for manipulating the Alzheimer dataset

Synthetic dataset from Rabie El Kharoua
https://www.kaggle.com/dsv/8668279

Code by Adrien Foucart 
(CC-BY-4.0 - https://creativecommons.org/licenses/by-nc/4.0/)


# Usage

dataset = AlzheimerDataset("alzheimers_disease_data.csv") # put the path to the CSV file here
# set a random state for replicability between your experiments
train_data, test_data = dataset.random_split(test_ratio=0.2, random_state=0)

# get a pandas.Series object with the requested column
values = train_data["Diabetes"]
# get the labels for the possible values
labels = train_data.labels("Diabetes")
# print the count per value for the Diabetes variable
print([(labels[v], (values==v).sum()) for v in np.unique(values)])
# note that this is a shorter way of writing:
for v in np.unique(values):
    print(labels[v], (values==v).sum()) 

# get a pandas.DataFrame with all nominal variables:
cats = data.nominals
# get a pandas.DataFrame with all the numerical variables:
nums = data.numericals
# get a pandas.DataFrame with all independent variables (all minus the ignored and outcome columns)
inds = data.independents
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Tuple, Dict


_DESCRIPTION = {
    'PatientID': {'type': 'ignore'},
    'Age': {'type': 'numerical'},
    'Gender': {'type': 'nominal',
               'labels': {0: 'Male', 1: 'Female', 2: 'X'}},
    'Ethnicity': {'type': 'nominal',
                  'labels': {0: 'Caucasian',
                             1: 'AfricanAmerican',
                             2: 'Asian',
                             3: 'Other'}},
    'EducationLevel': {'type': 'nominal',
                        'labels': {0: 'None',
                                   1: 'HighSchool',
                                   2: 'Bachelor',
                                   3: 'Higher'}},
    'BMI': {'type': 'numerical'},
    'Smoking': {'type': 'nominal', 'labels': {0: 'No', 1: 'Yes'}},
    'AlcoholConsumption': {'type': 'numerical'},
    'PhysicalActivity': {'type': 'numerical'},
    'DietQuality': {'type': 'numerical'},
    'SleepQuality': {'type': 'numerical'},
    'FamilyHistoryAlzheimers': {'type': 'nominal', 'labels': {0: 'No', 1: 'Yes'}},
    'CardiovascularDisease': {'type': 'nominal', 'labels': {0: 'No', 1: 'Yes'}},
    'Diabetes': {'type': 'nominal', 'labels': {0: 'No', 1: 'Yes'}},
    'Depression': {'type': 'nominal', 'labels': {0: 'No', 1: 'Yes'}},
    'HeadInjury': {'type': 'nominal', 'labels': {0: 'No', 1: 'Yes'}},
    'Hypertension': {'type': 'nominal', 'labels': {0: 'No', 1: 'Yes'}},
    'SystolicBP': {'type': 'numerical'},
    'DiastolicBP': {'type': 'numerical'},
    'CholesterolTotal': {'type': 'numerical'},
    'CholesterolLDL': {'type': 'numerical'},
    'CholesterolHDL': {'type': 'numerical'},
    'CholesterolTriglycerides': {'type': 'numerical'},
    'MMSE': {'type': 'numerical'},
    'FunctionalAssessment': {'type': 'numerical'},
    'MemoryComplaints': {'type': 'nominal', 'labels': {0: 'No', 1: 'Yes'}},
    'BehavioralProblems': {'type': 'nominal', 'labels': {0: 'No', 1: 'Yes'}},
    'ADL': {'type': 'numerical'},
    'Confusion': {'type': 'nominal', 'labels': {0: 'No', 1: 'Yes'}},
    'Disorientation': {'type': 'nominal', 'labels': {0: 'No', 1: 'Yes'}},
    'PersonalityChanges': {'type': 'nominal', 'labels': {0: 'No', 1: 'Yes'}},
    'DifficultyCompletingTasks': {'type': 'nominal', 'labels': {0: 'No', 1: 'Yes'}},
    'Forgetfulness': {'type': 'nominal', 'labels': {0: 'No', 1: 'Yes'}},
    'Diagnosis': {'type': 'outcome', 'labels': {0: 'No', 1: 'Yes'}},
    'DoctorInCharge': {'type': 'ignore'}
}


class AlzheimerData:
    """Helper class to manipulate Alzheimer data.

    # Example
    dataset = AlzheimerDataset("dataset.csv") # load using AlzheimerDataset class
    data = dataset.data # get all data
    v = data["ADL"] # get all values for the ADL variable, as a pd.Series object
    print(data.columns) # print the name of all the variables
    nums = data.numericals # get a pandas.DataFrame with all numerical variables
    cats = data.nominals # get a pandas.DataFrame with all nominal variables
    inds = data.independents # get a pandas.DataFrame with all variables except the outcome variable
    out = data.outcome # get a pandas.DataFrame with the outcome variable (Diagnosis)
    """
    def __init__(self, data: pd.DataFrame):
        self.dataframe = data
        self._nominals = {k: v for k, v in _DESCRIPTION.items() if v['type'] == 'nominal'}
        self._numericals = {k: v for k, v in _DESCRIPTION.items() if v['type'] == 'numerical'}
        self._outcome = {k: v for k, v in _DESCRIPTION.items() if v['type'] == 'outcome'}
        self._ignore = {k: v for k, v in _DESCRIPTION.items() if v['type'] == 'ignore'}

    def describe(self, ostream = print) -> None:
        ostream("# Alzheimer Dataset")
        ostream("")
        ostream("## Nominal variables:")
        for k, v in self._nominals.items():
            ostream(k)
            ostream("   Possible values:", v['labels'].values())
        ostream("## Numerical variables")
        for k, v in self._numericals.items():
            ostream(k)
        ostream("## Outcome:")
        for k, v in self._outcome.items():
            ostream(k)
            ostream("   Possible values:", v['labels'].values())

    def __getitem__(self, c: str | List[str]) -> pd.Series:
        return self.dataframe[c]

    @property
    def columns(self) -> pd.Index:
        return self.dataframe.columns

    @property
    def numericals(self) -> pd.DataFrame:
        return self.dataframe[self._numericals.keys()]

    @property
    def nominals(self) -> pd.DataFrame:
        return self.dataframe[self._nominals.keys()]

    @property
    def independent(self) -> pd.DataFrame:
        return self.dataframe[list(self._numericals.keys())+list(self._nominals.keys())]

    @property
    def outcome(self) -> pd.DataFrame:
        return self.dataframe[self._outcome.keys()]

    def labels(self, c) -> Dict[int, str]:
        if 'labels' not in _DESCRIPTION[c]:
            raise ValueError(f"{c} doesn't have label, it is a variable of type {_DESCRIPTION[c]['type']}")
        return _DESCRIPTION[c]['labels']


class AlzheimerDataset:
    """Helper class to manipulate the Alzheimer's dataset and split a train and test set
    
    Example:

    dataset = AlzheimerDataset("dataset.csv")
    data = dataset.data # returns a AlzheimerData object with all the data
    train, test = dataset.random_split() # returns a tuple of AlzheimerData with the train and test sets
    """

    def __init__(self, path: pathlib.Path | str):
        self.dataset = pd.read_csv(path)

        self.train_idxs = None
        self.test_idxs = None

    
    def random_split(self, test_ratio: float = 0.2, random_state: int | None = None) -> Tuple[AlzheimerData, AlzheimerData]:
        idxs = np.arange(len(self.dataset.index))
        np.random.seed(random_state)
        np.random.shuffle(idxs)
        self.test_idxs = idxs[:int(test_ratio*len(idxs))]
        self.train_idxs = idxs[int(test_ratio*len(idxs)):]

        train_data = self.dataset.iloc[self.train_idxs]
        test_data = self.dataset.iloc[self.test_idxs]

        return AlzheimerData(train_data), AlzheimerData(test_data)

    @property
    def data(self) -> AlzheimerData:
        return AlzheimerData(self.dataset)


def _test_alzheimer_data(path: str):
    dataset = AlzheimerDataset(path)
    train_data, test_data = dataset.random_split(test_ratio=0.2, random_state=0)

    values = train_data["Diabetes"]
    # get the labels for the possible values
    labels = train_data.labels("Diabetes")
    # print the count per value for the Diabetes variable
    print([(labels[v], (values==v).sum()) for v in np.unique(values)])
    # note that this is a shorter way of writing:
    for v in np.unique(values):
        print(labels[v], (values==v).sum()) 

    # get a pandas.DataFrame with all nominal variables:
    cats = train_data.nominals
    print(cats)
    # get a pandas.DataFrame with all the numerical variables:
    nums = train_data.numericals
    print(nums)

if __name__ == "__main__":
    # Test that we can load the data, specified in argument
    import sys
    if len(sys.argv) < 2:
        print(f"Usage: python alzheimer.py path/to/alzheimer_disease_data.csv")
    else:
        _test_alzheimer_data(sys.argv[1])

