# Author: Marius Lindauer
# License: BSD

import logging
logging.basicConfig(level="DEBUG")
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from aslib_scenario.aslib_scenario import ASlibScenario
from validation.validate import Validator

if __name__ == "__main__":

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--result_fn", help="Result json file with predictions for each test instances")
    parser.add_argument("--test_as", help="Directory with *all* test data in ASlib format")
    
    args_ = parser.parse_args()
    
    #read scenario
    scenario = ASlibScenario()
    scenario.read_scenario(dn=args_.test_as)
    
    # read result file
    with open(args_.result_fn) as fp:
        schedules = json.load(fp)
    
    validator = Validator()
    
    if scenario.performance_type[0] == "runtime":
        validator.validate_runtime(schedules=schedules, test_scenario=scenario)
    else:
        validator.validate_quality(schedules=schedules, test_scenario=scenario)