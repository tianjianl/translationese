import pandas ad pd

def data_to_df(task, langauge, split):
    
    if task == "xnli":
        filename = f"../xnli/{split}-{language}.tsv"
        f = open(filename, 'r')
        source = []
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

        df = pd.DataFrame({"src": src, "label": label})
        print(df.head(5)) 
        return df
    else:
        return None
