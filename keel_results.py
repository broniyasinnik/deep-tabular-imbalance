import glob
import re
import json
import yaml
import pandas as pd

def process_keel_results(root: str):
    result_files = glob.glob(f'{root}/*/logs/*/results/metrics.json')
    pattern = re.compile('\./Keel1/(?P<data_name>.*?)/logs/(?P<experiment_name>.*?)/results/metrics\.json')
    records = []
    for path in result_files:
        m = re.match(pattern, path)
        data_name = m.group('data_name')
        exper_name = m.group('experiment_name')
        with open(path) as f:
            res = json.load(f)
            records.append((data_name, exper_name, res['average_precision']))

    df = pd.DataFrame.from_records(records, columns=['dataset', 'experiment', 'ap'])
    df = df.pivot(index='dataset', columns='experiment', values='ap')
    ir = get_ir_in_configs(root)
    df["ir"] = pd.Series(ir)
    return df


def get_ir_in_configs(root: str):
    files = glob.glob(f'{root}/*/config.yml')
    pattern = re.compile('\./Keel1/(?P<data_name>.*?)/config\.yml')
    d = dict()
    for file in files:
        m = re.match(pattern, file)
        data_name = m.group('data_name')
        with open(file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        d[data_name] = config['ir']
    return d

if __name__ == "__main__":
    root = './Keel1'
    df = process_keel_results(root)
    # df.to_csv('./results_last.csv')