# Mixture of Experts

## Introduction

This is a basic implementation of the [paper](https://reader.elsevier.com/reader/sd/pii/S089360809900043X?token=B1AC91E547A7C1A099D743421A8BDC740112CA6B19B5B59EEC380BE54DB1B19CE8AF89A5291B2D20A358DF903DA798BD) and basically is a toy implementation of the Mixture of Experts algorithm.

So the model basically consist of various expert models which specialize at a particular task rather than a single model being good at that task.
And finally weights are assigned to the various experts using a gating network(kind of like attention) where more weight, as a result, is given to the expert good at the particular task in hand.

## Running the code

For training the model

- Clone the repository and go to the repo.
```
python main.py --training True    ### For training

python main.py --testing True     ### For testing
```

- Apart from this, the various hyperparameter flags can also be seen from the `main.py` file and can be tweaked accordingly.

## Code structure

- `main.py`: Specification of various hyperparameters used during training, along with checkpoint location specifications.

- `train.py`: Script for training(along with validating) the model and contains the whole training procedure.

- `test.py`: Script for testing the already trained model.

- `model.py`: Contains the architecture of model and the backbone used.

- `utils.py`: Contains the various helper functions along with function for getting dataset.

## Further things to be done

- I am still not able to completely get the EM algorithm specified in the paper for optimizing the weights, the reason for which has also been specified in the `utils.py` file.