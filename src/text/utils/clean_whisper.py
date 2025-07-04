#pip install nltk

from nltk.util import ngrams #for creating n-grams
from textwrap import wrap    #for separateing words into n-characters long pieces
import unicodedata           #for finding rare punctuation
import copy                  #for making a capy of the json without changing the original (yes, it was a problem)
import re                    #for finding instances of repetitions in texts
from glob import glob
import argparse
import pandas as pd
#DEFAULT ARGUMENTS FOR FUNCTIONS
#Motivation: since the functions call one another, it allows for easier argument transfer down the line
#Process: user arguments are passed with argument_name = argument_value, if no argument is passed then default
defaults = {"punctuation_token": "ⓟ",  #a token to replace all punctuation, should not appear in the text otherwise!
            "replacement_token": "ⓣ",  #a token to replace all repetitions, should not appear in the text otherwise!
            "color": "red",             #the color for showcasing repetitions
            "allow_repetitions": 2,     #this or less repetitions are allowed to occur in text
                                        #inside merged repetitions, will always be at least 3
                                        #allows for words that have 3 repeated letters/syllables inside
            "suspicious_length": 10,    #words longer that this will be checked for repetitions inside
            "n_grams_span": (1, 50)}    #the expected lowers and highest amount of words in repetitions
                                        #flexibly changes later if repetitions close to the highest amount are found

#AVAILABLE COLORS
#Motivation: pretty
colors = {"red":'\033[31m', "green": '\033[32m', "yellow": '\033[33m', "blue": '\033[34m', 
          "magenta": '\033[35m', "cyan": '\033[36m', "underline": '\033[4m', "bold": '\033[1m'}
end_chr, end_len = '\033[0m', 7



#PREPROCESSING TEXT
#Motivation: to find repetition even though they are capitalized/have different punctuation sign
#Process: lowers the given text and replaces every character marked as punctuation in unicode with the punctuation token
#Asssosiated parameters: punctuation_token
#Example: "One, one, one." -> "oneⓟ oneⓟ oneⓟ"
def preprocess(text: str,  **kwargs) -> str:
    kwargs = {**defaults, **kwargs}
    try: punctuation_token = str(kwargs["punctuation_token"])
    except Exception: raise TypeError("punctuation_token can only be a string!")
    
    preprocessed_text = text.lower()
    for i, character in enumerate(preprocessed_text):
        #all categories for punctuation in unicode begin with P
        if unicodedata.category(character).startswith('P'): 
            #replace() does not work with some rare characters
            preprocessed_text = preprocessed_text[:i] + punctuation_token + preprocessed_text[i+1:] 
            
    return preprocessed_text



#FINDING REPETITIONS INSIDE WORDS
#Motivation: to find repetitions merged into one word
#Process: finds words longer than suspicious_length, separates them into n-characters long pieces
         #checks if the pieces match each other, adds them to repetitions, repeats with a n+1
         #all instances of found_repetition*3 are removed from the local text to avoid later detection
         #for "lalalalalalalala", if "la" was already found, we do not need to find "lala"
#Asssosiated parameters: suspicious_length, allow_repetitions, preprocess() parameters
#Example: "AaaaaaabbbbCDCDCDCD. eeeee" -> ["a", "b", "cd"]  #no "e" because the len("eeeee")<suspicious_length
def find_merged_repetitions(text: str, **kwargs) -> list:
    kwargs = {**defaults, **kwargs}
    try: suspicious_length = int(kwargs["suspicious_length"])
    except Exception: raise TypeError("suspicious_length can only be an integer!")
    try: allow_repetitions = kwargs["allow_repetitions"]
    except Exception: raise TypeError("allow_repetitions can only be an integer!")
    if allow_repetitions < 4: allow_repetitions = 4 #always >3 so words with triple of the same letter (e.g. "Schifffahrt") are not detected
    text = preprocess(text, **kwargs)
    
    #a utility function
    def check_for_repetitions(long_word):
        repetitions = []
        #splitting the word into n-character-long pieces
        for n in range(1, len(long_word)//allow_repetitions):
            parts = wrap(long_word, n)
            #if a few (allow_repetitions) pieces in a row are the same, adding them to repetitions
            for j in range(len(parts)-allow_repetitions):                    
                if parts[j] not in repetitions and len(set([parts[j+k] for k in range(allow_repetitions)]))==1:
                    repetitions.append(parts[j])
                    #removing chains of repetitions locally to avoid later detection
                    long_word=long_word.replace(parts[j]*3, "")
                    
        return(repetitions)
        
    #the primary function continues
    repetitions = []
    #only checking words longer that suspicious_length
    words = [word for word in text.split() if len(word)>suspicious_length]
    for word in words:
        repetition = check_for_repetitions(word)
        repetitions.extend([i for i in repetition if i not in repetitions])
        
    return repetitions



#FINDING REPETITIONS THAT ARE WORDS
#Motivation: to find repetitions sequences of one or more words
#Process: separates text into n-grams, checks if the n-grams match each other, adds them to repetitions, repeats with a n+1
         #checks up to n_max words at a time, or more if it finds repetitions
         #due to the nature of n-grams, the text "one two three one two three" would be separated into the following trigrams:
         #[("one", "two", "three"), ("two", "three", "one"), ("three", "one", "two"), ("one", "two", "three")]
         #n-grams i+n*j are checked matching: i = current index, n = size of the current n-gram, j in range [0, 1, ..., allowed repetitions-1]
         #if repetitions are found, i+n*j+k (for k in range [0, 1, ..., n-1]) are considered the shifts of the repetition and are skipped
         #e.g., for the repetition ("one", "two", "three"), ("two", "three", "one") and ("three", "one", "two") are the shifts
         #all instances of found_repetition*2 are removed from the local text to avoid later detection
         #for "la la la la la la", if "la" was already found, we do not need to find "la la"
#Asssosiated parameters: n_grams_span, allow_repetitions, preprocess() parameters
#Example: "One, two, one, two, one, two." -> ["oneⓟ, twoⓟ"] 
def find_spaced_repetitions(text=str, **kwargs) -> list:
    kwargs = {**defaults, **kwargs}
    n, n_max = kwargs["n_grams_span"]
    if type(n) != int or type(n_max) != int or n_max-n < 0:
          raise TypeError("n_gram_span can only be an integer!")
    allow_repetitions = kwargs["allow_repetitions"]
    text = preprocess(text, **kwargs)
    
    repetitions = []
    #n-gram sizes of >50< or >(biggest found repetition) * 2<
    while n < n_max or (len(repetitions) > 0 and n < len(repetitions[-1].split()) * 2):
        #spliting text n_grams, adding padding at the end
        n_grams = list(ngrams(text.split(), n))+n*[""]
        #since we are comparing to future n_grams, cut_off allows us to not go out of bounds with indices
        cut_off = n*allow_repetitions

        #restarting the algorythm to avoid double or cross detection of repetitions
        restart = True
        while len(n_grams) > cut_off and restart:     
            border = len(n_grams)-cut_off

            for i in range(border):
                n_gram_list = [n_grams[i+j*n] for j in range(allow_repetitions)]
                    
                #skipping repetitions that are already recorded plus all their shifts
                if " ".join(n_gram_list[0]) in repetitions:
                        n_grams = n_grams[i+cut_off-n:]
                        break
                #if a few (allow_repetitions) n-grams in a row are the same, adding them to repetitions
                #skipping the repetition shifts
                elif len(set(n_gram_list)) == 1:                                 
                    n_gram = " ".join(n_gram_list[0])
                    repetitions.append(n_gram)
                    n_grams = n_grams[i+cut_off-n:]
                    #removing chains of repetitions locally to avoid later detection
                    text = text.replace(n_gram+" "+n_gram, "")   
                    break
                #once the last n-gram is reached, we can move to the next n-gram size
                if i == border-1:
                    restart = False
        n+=1
        
    return repetitions



#FINDING ALL REPETITIONS
#Motivation: to make it easy to find all repetitions at once
#Process: returns the combined list from two previous functions
#Asssosiated parameters: find_spaced_repetitions() parameters, find_merged_repetitions() parameters
#Example: "One, two, one, two, one, two." -> ["oneⓟ, twoⓟ"] 
def find_repetitions(text=str, **kwargs) -> list:
    repetitions = find_spaced_repetitions(text, **kwargs)+find_merged_repetitions(text, **kwargs)
    return repetitions


    
#FIND THE START END END OF EACH REPETITION
#Motivation: to find the starts and ends of all repetition clusters that are longer than allow_repetitions
#Process: for each repetition, finds the spans of it occuring in the text
         #if the spans follow one another, they are merged into a bigger span
         #the end results spans are bigger then allow_repetitions*len(repetition)
         #repetitions and their end result spans are added to the dictionary as a key-value pair if the spans arent empty
         #for repetitions < 4 characters long occuring inside words, there exists a lot of single occurences, which slow down the process
         #in such cases, the model looks for the pairs of such repetitions in a row and then adds the odd last repetition to the span if such exists
#Asssosiated parameters: allow_repetitions, repetitions (makes new if none is given), find_repetitions() parameters
#Example: "Oneeeeeeeeee. Two, two, two. Threeeeeeeeeeeeee!" -> {'twoⓟ': [(14, 28)], 'e': [(2, 12), (32, 46)]}
def find_spans(text: str, repetitions: list=None, **kwargs) -> dict:
    kwargs = {**defaults, **kwargs}
    try: allow_repetitions = kwargs["allow_repetitions"]
    except Exception: raise TypeError("allow_repetitions can only be an integer!")
    text = preprocess(text, **kwargs)
    if repetitions == None: repetitions = find_repetitions(text, **kwargs)
        
    spans = {}
    
    #finding all instances of a repetition occuring
    for rep in repetitions:
        #helps speed up short string checking
        if len(rep) < 4 and " "+rep+" "+rep not in text:
            rep_spans = [i.span() for i in re.finditer(rep*2, text)]
            span_starts = [i[0] for i in rep_spans]
            rep_spans = [(span[0], span[1]+len(rep)) if span[1] not in span_starts and text[span[1]:span[1]+len(rep)]==rep else span for span in rep_spans]
        else:
            rep_spans = [i.span() for i in re.finditer(rep, text)]
            
        #restarting the algorithm after merging a span to see the bigger picture
        for restart in range(len(rep_spans)+1):
            #checking if a span and a span after it come right after another
            for i in range(len(rep_spans)-1):
                rs = sorted(rep_spans)
                first, second = rs[i], rs[i+1]
                first_start, first_end = first[0], first[1]
                second_start, second_end = second[0], second[1]
                follow_condition = first_end == second_start or first_end+1 == second_start
                if follow_condition:
                    #if two spans follow each other, merge them and remove the original ones
                    rep_spans.append((first_start, second_end))
                    rep_spans.remove(first)
                    rep_spans.remove(second)
                    break
             
        #checking that the found spans are longer than the allowed repetitions amount
        rep_span = [span for span in rep_spans if (span[1]-span[0])//len(rep)>allow_repetitions]
        #assigning valid spans to the repetitions
        if len(rep_span) !=0: spans[rep] = rep_span
                
    return spans


#REPLACE REPETITIONS WITH TOKENS
#Motivation: to mark down the parts of the repetitions to be deleted but keep the original length of the text so indices still work
#Process: for each repetition span, only preserve the first repetition sans the last character as well as the last character of the last repetition
         #if a repetition is one character long, only the first character
#Asssosiated parameters: replacement_token, repetitions_spans (makes new if none is given), find_spans() parameters
#Example: original: "Oneeeeeeeeee. Two, two, two. Three!" 
         #spans {'twoⓟ': [(14, 28)], 'e': [(2, 12)]}
         #padded: "Oneⓣⓣⓣⓣⓣⓣⓣⓣⓣ. Twoⓣⓣⓣⓣⓣⓣⓣⓣⓣⓣ. Three!" 
def pad_text(text:str, repetitions_spans: list=None, **kwargs) -> str:
    kwargs = {**defaults, **kwargs}
    try:replacement_token = kwargs["replacement_token"]
    except Exception: raise TypeError("replacement_token can only be a string!")
    if repetitions_spans == None: repetitions_spans = find_spans(text, **kwargs)

    #replacing all spans with token characters while only leaving one instance
    for rep, spans in repetitions_spans.items():
        for span in spans:
            start = span[0]+1 #preserving the first character in case its a capital letter
            end = span[1]-1   #precerving the last character in case its puntuation
            padding = replacement_token*(end-start-len(rep)+2)
            if len(rep) == 1: text = text[:start]+padding+text[end+1:]
            elif len(rep) == 2: text = text[:start]+padding+text[end:]
            else: text = text[:start+len(rep)-2]+padding+text[end:] #trust me bro

    return text



#REMOVE REPETITION TOKENS
#Motivation: to clean out the repetition tokens from the padded text
#Process: replace repetitions tokens with ""
#Asssosiated parameters: replacement_token, pad_text() parameters
#Example: original: "Oneeeeeeeeee. Two, two, two. Three!" 
         #padded: "Oneⓣⓣⓣⓣⓣⓣⓣⓣⓣ. Twoⓣⓣⓣⓣⓣⓣⓣⓣⓣⓣ. Three!" 
         #cleaned: "One. Two. Three!" 
def clean_text(text: str, **kwargs) -> str:
    kwargs = {**defaults, **kwargs}
    try:replacement_token = kwargs["replacement_token"]
    except Exception: raise TypeError("replacement_token can only be a string!")

    #removing the padding
    text = pad_text(text, **kwargs)
    text = text.replace(replacement_token, "")

    return text



#UTILITY FUNCTION FOR clean_segments()
#Motivation: to match the words from the cleaned sequence to the original words
#Process: every word in the original sequence 1) gets ignored if it is entirely or partially padded from the beginning in the padded sequence (repetition 2+)
                                             #2) gets replaced with the word from the cleaned sequence if only the last character is padded (repetition 1)
                                             #3) gets kept unchanged if it was not padded
         #returns the lists of matching words and indices of the words that should stay
#Asssosiated parameters: given by the clean_segments() function
#Example: original: "Oneeeeeeeeee. Two, two, two. Three!" 
         #padded: "Oneⓣⓣⓣⓣⓣⓣⓣⓣⓣ. Twoⓣⓣⓣⓣⓣⓣⓣⓣⓣⓣ. Three!" 
         #cleaned: "One. Two. Three!" 
         #returns: ["One.", "Two.", "Three!"], [0, 1, 4]
def clean_words_inside_segments(original: str, padded: str, cleaned:str, token: str) -> (list, list):    
    words_to_stay = []
    indices_to_stay = []
    regression_index, start = 0, 0
    for index, word in enumerate(original.split()):
        padded_word = padded[start: start+len(word)]
        if padded_word[0] == token: regression_index+=1
        elif token in padded_word:
            new_word = cleaned.split()[index-regression_index]
            words_to_stay.append(new_word)
            indices_to_stay.append(index)
        else: 
            words_to_stay.append(word)
            indices_to_stay.append(index)
                            
        start += len(word)+1
    return (words_to_stay, indices_to_stay)



#REMOVE REPETITIONS FROM JSON
#Motivation: to clean the json files, keep and adjust id, start, end, text and words columns
#Process: making a deepcopy of the original json to not change it in the future
         #merging the texts together and padding the resulting sequence
         #finding index spans for every segment and placing the padded text back accorning to those spans, removing tokens
         #cleanind out the words that are no longer present in the segments from "words"
         #adjusting the id distribution so there are no gaps between segments
#Asssosiated parameters: replacement_token, pad_text() parameters
#Example: original: [{id: 1, "text": "Zerooooooooooooooooooooooooooooo. Oneeeeeeee. Two, two, two. Three!", words: [{"text": "Zerooooooooooooooooooooooooooooo.", "n": 0}, {"text": "Oneeeeeeee.", "n": 1}, {"text": "Two,", "n": 2}, {"text": "two,", "n": 3}, {"text": "two.", "n": 4}, {"text": "Three!", "n": 5}]}]
         #returns: [{id: 0, "text": "Zero. One. Two. Three!", words: [{'text': 'Zero.', 'n': 0}, {'text': 'One.', 'n': 1}, {'text': 'Two.', 'n': 2}, {'text': 'Three!', 'n': 5}]}]
def clean_segments(segments: list, **kwargs) -> list:
    kwargs = {**defaults, **kwargs}
    try:replacement_token = kwargs["replacement_token"]
    except Exception: raise TypeError("replacement_token can only be a string!")
    
    text = "".join([line["text"] for line in segments])
    text = pad_text(text, **kwargs)
    try:
        new_segments = copy.deepcopy([{"id": i["id"], "start": i["start"], "end": i["end"], "text": i["text"], "words": i["words"], "confidence": i["confidence"]} for i in segments])
    except Exception: new_segments = copy.deepcopy(segments)
        
    #determining the spans of serments in the original text
    characters_passed = 0
    for n, segment in enumerate(new_segments):
        original_segment_text = segment["text"]
        currect_last_character = characters_passed+len(segment["text"])
        segment_span = (characters_passed, currect_last_character)
        #extracting the string matching the span
        proposed_segment_text_padded = text[segment_span[0]:segment_span[1]]
        proposed_segment_text_cleaned = proposed_segment_text_padded.replace(replacement_token, "")
        characters_passed = currect_last_character

        if original_segment_text != proposed_segment_text_cleaned:
            #assigning the new text to the segment
            segment["text"] = proposed_segment_text_cleaned
            #cleaning up the "words" part and assigning it to the segment
            if len(proposed_segment_text_cleaned) != 0 and "words" in segment: 
                words_to_stay, indices_to_stay = clean_words_inside_segments(original_segment_text,
                                                                             proposed_segment_text_padded, 
                                                                             proposed_segment_text_cleaned,
                                                                             replacement_token)
                segment["words"] = [segment["words"][i] for i in indices_to_stay]
                for i in range(len(segment["words"])):
                    segment["words"][i]["text"] = words_to_stay[i]

    #deleting segments with no text
    new_segments = [i for i in new_segments if i["text"]!=""]
    #reassigning the id numbers to avoid spaces
    for i in range(len(new_segments)): new_segments[i]["id"] = i

    return new_segments


#CLEAN TEXT OR JSON
#Motivation: to not always have to search for an appropriate function for your data type
#Process: detects object type and chooses the appropriate cleaning function
#Asssosiated parameters: clean_text() parameters, clean_segments() parameters
def clean(obj, **kwargs):
    if type(obj) == str: return clean_text(obj, **kwargs)
    
    if type(obj) == list: 
        if type(obj[0]) == dict: return clean_segments(obj, **kwargs) 
        
    if type(obj) == dict:
        if "text" in obj and "segments" in obj: 
            return {"text":clean_text(obj["text"], **kwargs), "segments": clean_segments(obj["segments"], **kwargs)}
            
    #if nothing got returned
    raise TypeError("Only strings or jsons are allowed")



#SHOW THE REPETITIONS INSIDE A TEXT
#Motivation: to clearly see what repetitions are found and where
#Process: calculate and show the repetition percentage, the found repetitions, and the text where the color tokens are placed at the beginning and end of each repetition span
#Asssosiated parameters: color, repetitions_spans() parameters, clean_text() parameters
def showcase(text: str, **kwargs) -> None:
    kwargs = {**defaults, **kwargs}
    try:replacement_token = kwargs["replacement_token"]
    except Exception: raise TypeError("replacement_token can only be a string!")
    try:punctuation_token = kwargs["punctuation_token"]
    except Exception: raise TypeError("punctuation_token can only be a string!")
    
    color = kwargs["color"]
    if color in colors: color_chr = colors[color]
    else: 
        print(f"The color {color} is not available, repetitions are highlighted in red.")
        color, color_chr = "red", colors["red"]
    col_len = len(color_chr)
    repetitions_spans = find_spans(text, **kwargs)
    cleaned_text = clean_text(text, **kwargs)
    repetition_percent = round(100-len(cleaned_text)/len(text)*100, 2)

    marked_text = text
    index_shift=0
    spans = []
    for i in sorted(repetitions_spans.values()):
        spans.extend(i)
    for span in spans:
        start = span[0]+index_shift
        end = span[1]+index_shift
        marked_text = marked_text[:start]+color_chr+marked_text[start:end]+end_chr+marked_text[end:]
        index_shift += col_len+end_len-3
    
    print(f"This text contains {repetition_percent}% repetitions. The following repetitions were found:")
    if repetitions_spans != {}:
        print(' | '.join([color_chr+i+end_chr for i in repetitions_spans.keys()]))
    print()
    print(marked_text)

def rw_text_file(source_path):
    """
    Reads all text from a source file and writes it to a destination file.
    
    Args:
        source_path (str): Path to the source text file
        destination_path (str): Path to the destination text file
    """
    destination_path = str.replace(source_path, ".txt", "") + "cleaned.txt"
    try:
        # Read all text from the source file
        with open(source_path, 'r', encoding='utf-8') as source_file:
            file_content = source_file.read()
        
        file_content = clean(file_content)

        # Write the content to the destination file
        with open(destination_path, 'w', encoding='utf-8') as destination_file:
            destination_file.write(file_content)
            
        print(f"Successfully copied content from {source_path} to {destination_path}")
    
    except FileNotFoundError:
        print(f"Error: The file {source_path} was not found.")
    except IOError as e:
        print(f"An error occurred while handling the files: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def process_csv(input_file, output_file):
    """
    Process CSV file: clean 'sentence' column and save with new column
    """
    # Read CSV file
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found!")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Verify required columns exist
    required_columns = ['dataset', 'class', 'file', 'sentence']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        print(f"Missing columns in CSV: {missing}")
        return

    # Apply cleaning function to 'sentence' column
    df['sentences_clean'] = df['sentence'].apply(clean)

    # Write back to CSV
    try:
        df.to_csv(output_file, index=False)
        print(f"Successfully processed file. Output saved to: {output_file}")
    except Exception as e:
        print(f"Error writing CSV: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--csv_folder", type=str)
    args = parser.parse_args()
    folder_path = args.csv_folder
    csvs = glob(folder_path+"/*.csv")
    # jsons = glob(folder_path+"*.json")
    for c in csvs:
        print(c)
        process_csv(c, str.replace(c, ".csv", "_cleaned.csv"))