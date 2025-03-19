import numpy as np
import pysrt
import os
import nltk
from pandas.io.common import file_path_to_url
from sklearn.metrics import confusion_matrix
import json
from rouge_score import rouge_scorer
def srt_time_to_seconds(srt_time):
    # Split the time string by colon and comma
    try:
        hours, minutes, seconds, milliseconds  = srt_time.hours, srt_time.minutes, srt_time.seconds, srt_time.milliseconds
        # Convert each part to an integer
        hours = int(hours)
        minutes = int(minutes)
        seconds = int(seconds)
        milliseconds = 0 #float(milliseconds)

        # Calculate the total seconds
        total_seconds = (
            hours * 3600 +      # Convert hours to seconds
            minutes * 60 +      # Convert minutes to seconds
            seconds +           # Add seconds
            milliseconds / 1000 # Convert milliseconds to seconds
        )
        return int(total_seconds)
    except ValueError:
        raise ValueError(f"Invalid SRT time format: {srt_time}")

def seconds_to_timestamp(seconds):
    # Convert seconds to hours, minutes, and seconds
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)

    # Format the timestamp
    timestamp = f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"
    return timestamp

def read_srt(input_file_path):
    if not input_file_path.lower().endswith('.srt'):
        print("The provided file does not have an .srt extension.")
        # Load in the .srt file
    try:
        subs = pysrt.open(input_file_path)
    except Exception as e:
        print(f"Error reading the SRT file: {e}. Returning an empty dummy file")
        subs = pysrt.SubRipFile()
        sub = pysrt.SubRipItem(1, start='00:00:04,000', end='00:02:08,000', text="Hello World!")
        subs.append(sub)
    return subs

def write_logs(out_folder, predictions,times, eval_metrics, mode = ""):

    out_file = os.path.join(out_folder, f'{mode}.json')
    with open(out_file, 'w') as f:
        json.dump(eval_metrics, f)

    out_file = os.path.join(out_folder, f"logs_{mode}.txt")
    print(f"Generation stored at {out_file}")
    with open(out_file, 'a') as the_file:
        for t, ut in zip(times, predictions):
            ut = ut.replace("\n", "")
            the_file.write(f"{t}: {ut}\n")
    return out_file
def get_commentary_path(commentary_directory, game_path):
    game_path = os.path.basename(game_path)
    commentary_path = [os.path.join(commentary_directory, file) for file in os.listdir(commentary_directory) if
     file.endswith('.srt') and os.path.isfile(os.path.join(commentary_directory, file)) and "kyakkan" in file
         and game_path in file]
    if len(commentary_path) > 0:
        commentary_path = commentary_path[0]
    else:
        commentary_path = None
    return commentary_path


def remove_repeatitions(utterences):
    pred_utterences_cleaned = []
    previous = ""
    for pred_utterence in utterences:
        if previous != pred_utterence:
            pred_utterences_cleaned.append(pred_utterence)
        else:
            pred_utterences_cleaned.append("")
        previous = pred_utterence
    return pred_utterences_cleaned


def compute_metrics(ref_timing, pred_timing, pred_utterences, ref_utterences):
    correlations = [1 if a == b else 0 for a, b in zip(ref_timing, pred_timing)]
    cm = confusion_matrix(ref_timing, pred_timing)

    pred_commentary = " ".join(pred_utterences)
    ref_commentary = " ".join(ref_utterences)
    r_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge = r_scorer.score(ref_commentary, pred_commentary)

    BLEUscore = nltk.translate.bleu_score.sentence_bleu([ref_commentary], pred_commentary, weights=(0.5, 0.5))


    res =  {"correlation":(correlations.count(1))/len(correlations), "rouge": rouge, "blue": BLEUscore,  "ref_timing": list(ref_timing),
            "pred_timing": list(pred_timing)}
    return res


def estimate_talking_speed(sample_file):
    sample_commentary = read_srt(sample_file)
    words_per_second = []
    for utterence in sample_commentary:
        time_to_speak = float(srt_time_to_seconds(utterence.duration)) + 0.5
        num_of_words = len(utterence.text.split())
        words_per_second.append(float(num_of_words/time_to_speak))

    return np.mean(words_per_second)

def convert_text_to_srt(file_path: str = None, talking_speed_sample:str = "../RaceCommentary/transcriptions_whole_data_english/AC_120221-180622_R_ks_audi_r8_plus_ks_nurburgring_layout_sprint_a_kyakkan.merged.mp4_translated.srt" ):
    if file_path is None:
        print("r Filepath with timestamps should be provided")
        exit()
    seconds_per_word = estimate_talking_speed(
        sample_file=talking_speed_sample)
    if file_path is not None:
        timestamps = []
        utterances = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
        print (lines)
        for line in lines:
            l = line.split(':')
            t = int(l[0])
            ut = str(l[1]).strip()
            if ut and "WAIT" not in ut:
                num_of_words = len(ut.split())
                start_time = seconds_to_timestamp(t)
                end_time = seconds_to_timestamp(t+(num_of_words*seconds_per_word))
                timestamps.append((start_time, end_time))
                utterances.append(ut)
        # Define the filename for the .srt file
        srt_filename = file_path.replace('.txt','.srt')

        # Create the .srt file
        with open(srt_filename, 'w') as srt_file:
            for i, (start, end) in enumerate(timestamps):
                # Write sequence number
                srt_file.write(f"{i + 1}\n")
                # Write timestamp
                srt_file.write(f"{start} --> {end}\n")
                # Write utterance
                srt_file.write(f"{utterances[i]}\n")
                # Blank line to separate entries
                srt_file.write("\n")

        print(f"SRT file '{srt_filename}' created successfully.")





#convert_text_to_srt(
#    file_path="../logs/llava-hf_LLaVA-NeXT-Video-34B-hf/AC_100221-115136_R_ks_porsche_cayenne_mugello_/step_1_frames-used_1/logs_feedback_loop.txt")
