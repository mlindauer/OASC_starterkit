# OASC_starterkit

The Open Algorithm Selection Challenge (OASC) will take place in Summer 2017. 
To this end, we provide here some simple scripts
which allow newcomers to read the data into pandas
and show an example how to generate an output file in the required format.

## Installation

The example requires Python 3.4 or later.
To install the required dependencies please call:

``` pip install -r requirements.txt ```

(Most requirements are already satisfied by using Anaconda.)

## Example Usage

The example script simpy computes the single best algorithm across all training instances
and predicts it for each test instance.

To call the script, please call:

```python oasc_starterkit/single_best.py --train_as example_files/SAT11-INDU-TRAIN/ --test_as example_files/SAT11-INDU-TEST/```

Please have a look in the extensively documented code 
for some explanations.

In the end, the scripts writes the file `results.json` to disk
which could be submitted in this format to the competition.

## Contact

Marius Lindauer
lindauer@cs.uni-freiburg.de

