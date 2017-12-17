import pandas as pd
import os.path
import glob
import sys

def main(folder_name):
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, folder_name)
    extension = 'csv'
    os.chdir(path)
    csv_files = [i for i in glob.glob('*.{}'.format(extension))]

    df = pd.read_csv(csv_files[0])
    for file in csv_files[1:]:
        df = df.append(pd.read_csv(file))

    df = df.reset_index(drop=True)
    print(df)
    return df

if __name__ == "__main__":
    folder_name = sys.argv[1]
    main(folder_name)

