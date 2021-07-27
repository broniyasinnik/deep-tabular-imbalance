import glob
import re
import json
import yaml
import pandas as pd

def process_keel_results(root: str):
    result_files = glob.glob(f'{root}/*/results/*/metrics.csv')
    pattern = re.compile('\./Keel1/(?P<data_name>.*?)/results/(?P<experiment_name>.*?)/metrics\.csv')
    records = []
    result_df = None
    for path in result_files:
        m = re.match(pattern, path)
        assert m is not None, f"No matching result in {path}"
        data_name = m.group('data_name')
        exper_name = m.group('experiment_name')
        print(data_name, " ", exper_name)
        df = pd.read_csv(path)
        index = pd.MultiIndex.from_tuples([(data_name, exper_name)], names=["dataset", "experiment"])
        df1 = pd.DataFrame(df.to_numpy(), index=index, columns=df.columns)
        if result_df is None:
            result_df = df1
        else:
            result_df = pd.concat([result_df, df1])
    return result_df
    #     with open(path) as f:
    #         res = json.load(f)
    #         records.append((data_name, exper_name, res['average_precision']))
    #
    # df = pd.DataFrame.from_records(records, columns=['dataset', 'experiment', 'ap'])
    # df = df.pivot(index='dataset', columns='experiment', values='ap')
    # ir = get_ir_in_configs(root)
    # df["ir"] = pd.Series(ir)
    # return df


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