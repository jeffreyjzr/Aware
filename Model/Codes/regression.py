import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import argparse
import datetime
import numpy as np


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run multiple logistic regression.")

    parser.add_argument('--data', nargs='?', default='',
                        help='Input data path')
    parser.add_argument('--iter', type=int, default=100,
                        help='Input max iterations')
    parser.add_argument('--params', nargs='?', default='',
                        help='Output params path')
    parser.add_argument('--pvalues', nargs='?', default='',
                        help='Output pvalues path')
    parser.add_argument('--tests', nargs='?', default='',
                        help='Output tests path')
    return parser.parse_args()


def main(args):
    starttime = datetime.datetime.now()
    col_names = ['gender', 'age', 'occupation', 'education', 'area', 'ifhavebaby', 'ifmarried','purchase','proximity','family','schoolmate','workmate','label']
    
    # load dataset
    print("load data...")
    datasets = pd.read_csv(args.data, header=None, names=col_names)

    print("model...")
    f = 'label ~ C(gender) + age  + C(occupation,Treatment(1)) + C(education)  + C(ifhavebaby) + C(ifmarried) + purchase + proximity + family + schoolmate+ workmate'
    
    result = smf.mnlogit(formula=str(f), data=datasets).fit(maxiter=args.iter, method='newton', skip_hessian=True)
    name = ['Pseudo R^2','LLR', 'LLR P Value']
    
    print(result.summary())
   
    result.params.to_csv(args.params, sep='\t', header=False, index=True)
    result.pvalues.to_csv(args.pvalues, sep='\t', header=False, index=True)
    l = []
    l.append(result.prsquared)
    l.append(result.llr)
    l.append(result.llr_pvalue)
    tests = pd.DataFrame(index=name,data=l)
    tests.to_csv(args.tests, sep='\t', header=False, index=True)

    endtime = datetime.datetime.now()
    print('Running time: ' + str((endtime - starttime).seconds))


if __name__ == "__main__":
    args = parse_args()
    main(args)
