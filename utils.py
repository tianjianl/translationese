import pandas as pd

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
