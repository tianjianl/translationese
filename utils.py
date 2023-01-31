import pandas as pd
from glob import glob
def data_to_df(task, language, split):
    
    if task == "xnli":
        filename = f"../xnli/{split}-{language}.tsv"
        f = open(filename, 'r')
        src = []
        labels = []
        for line in f:
            cols = line.strip().split('\t')
            premise = cols[0]
            hypothesis = cols[1]
            if cols[2] == 'neutral':
                label = 0
            elif cols[2] == 'entailment':
                label = 1
            elif cols[2] == 'contradiction':
                label = 2
                
            src.append(premise + " " + hypothesis)
            labels.append(label)

        df = pd.DataFrame({"src": src, "label": labels})
        df = df.sample(frac=1, ignore_index=True)
        print(df.head(5)) 
        return df
    
    elif task == 'pawsx':
        filename = f"../pawsx/{split}-{language}.tsv"
        f = open(filename, 'r')
        src = []
        labels = []
        for line in f:
            cols = line.strip().split('\t')
            src.append(cols[0] + ' ' + cols[1])
            labels.append(int(cols[2]))
        
        df = pd.DataFrame({"src": src, "label": labels})
        df = df.sample(frac=1, ignore_index=True)
        print(df.head(5))
        return df
    
    elif task == 'ape':
        mt_filenames = glob('../ape/*/*.mt')
        pe_filenames = glob('../ape/*/*.pe')
        src = []
        labels = []
        for mt_file in mt_filenames:
            if split not in mt_file:
                continue
            f = open(mt_file, 'r')
            for line in f:
                line = line.strip()
                src.append(line)
                labels.append(0)

        for pe_file in pe_filenames:
            if split not in pe_file:
                continue
            f = open(pe_file, 'r')
            for line in f:
                line = line.strip()
                src.append(line)
                labels.append(1)

        df = pd.DataFrame({"src": src, "label": labels})
        df = df.sample(frac=1, ignore_index=True)
        print(df.head(5))
        return df.head(20000)
