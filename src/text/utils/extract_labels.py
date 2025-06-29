import pandas as pd
from .global_variables import EMOTIONS_EXT

def extract_labels(df: pd.DataFrame) -> list[list[int]]:
    one_hot_list = []
    for _, row in df.iterrows():
        vect8 = [  
            int(row[EMOTIONS_EXT[0]]), # Neutral
            int(row[EMOTIONS_EXT[1]]), # Anger
            int(row[EMOTIONS_EXT[2]]), # Disgust
            int(row[EMOTIONS_EXT[3]]), # Feat
            int(row[EMOTIONS_EXT[4]]), # Happiness
            int(row[EMOTIONS_EXT[5]]), # Sadness
            int(row[EMOTIONS_EXT[6]]), # Surprise
            int(row[EMOTIONS_EXT[7]]) # other
        ]
        one_hot_list.append(vect8)
    return one_hot_list