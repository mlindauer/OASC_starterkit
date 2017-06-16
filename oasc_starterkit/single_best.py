# Author: Marius Lindauer
# License: BSD 
# This is a very simple script to show 
# a baseline predictor can be implemented
# based on an ASlib Scenario

import os
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from aslib_scenario.aslib_scenario import ASlibScenario

class SingleBest(object):
    
    def __init__(self):
        self.single_best = None
    
    def main(self,
             train_scenario_dn:str,
             test_scenario_dn:str=None):
        '''
            main method
            
            Arguments
            ---------
            train_scenario_dn:str
                directory name with ASlib scenario training data
            test_scenarios_dn:str
                directory name with ASlib scenario test data 
                (performance data is missing)
        '''
        
        # Read scenario files
        scenario = ASlibScenario()
        scenario.read_scenario(dn=train_scenario_dn)
        
        # fit on training data
        self.fit(scenario=scenario)
        
        # Read test files
        # ASlibScenario is not designed to read partial scenarios
        # therefore, we have to cheat a bit
        scenario = ASlibScenario()
        scenario.read_description(fn=os.path.join(test_scenario_dn,"description.txt"))
        scenario.read_feature_values(fn=os.path.join(test_scenario_dn,"feature_values.arff"))
        scenario.read_feature_runstatus(fn=os.path.join(test_scenario_dn,"feature_runstatus.arff"))
        
        # predict on test data
        self.predict(scenario=scenario)

    def fit(self, scenario:ASlibScenario):
        '''
            fit an algorithm selector on data in scenario
            
            An Scenario object has fields such as the following pandas
            feature_data, performance_data, runstatus_data
            (see ASlibScenario code for more details)
            
            In this starter kit,
            we will compute only the single best solver

            Arguments
            ---------
            scenario: ASlibScenario
                scenario with training data
        '''
        
        # get performance data
        perf_data = scenario.performance_data
        
        # average performance for each algorithm
        average_perf = perf_data.mean(axis=0)
        print(average_perf)
        
        # get best performing algorithm
        # assumption minimization -- 
        # ASlibScenario automatically multiplies by -1 
        # if performance measure has to be maximized
        self.single_best = average_perf.argmin()
        print(self.single_best)
        
    def predict(self, scenario:ASlibScenario):
        '''
            select an algorithm for each instance in given scenario
            
            the scenario will have the pandas for 
            feature_data, feature_runstatus_data
        '''
        
        # get features
        features = scenario.feature_data
        
        #print("Test Features")
        #print(features)
        
        # we ignore the features in this example
        # and only iterate over all instances
        # and build result dictionary
        predictions = {}
        for inst_ in features.index:
            # always predict single best algorithm
            # as a show case, distinguish between runtime and quality
            if scenario.performance_type[0] == "runtime":
                predictions[inst_] = [(self.single_best,scenario.algorithm_cutoff_time)]
            elif scenario.performance_type[0] == "solution_quality":
                predictions[inst_] = [(self.single_best,999999999999)]
                
        # dump results to disk
        # overwrites old results!
        with open("results.json", "w") as fp:
            json.dump(predictions, fp=fp, indent=2)
                
if __name__ == "__main__":

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_as", help="Directory with training data in ASlib format")
    parser.add_argument("--test_as", help="Directory with test data in ASlib format")
    
    args_ = parser.parse_args()
    
    sb = SingleBest()
    sb.main(train_scenario_dn=args_.train_as,
            test_scenario_dn=args_.test_as)
        