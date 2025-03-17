import pysrt
import os
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
    print(eval_metrics)
    with open(out_file, 'w') as f:
        json.dump(eval_metrics, f)

    out_file = os.path.join(out_folder, f"logs_{mode}.txt")
    print(f"Generation stored at {out_file}")
    with open(out_file, 'a') as the_file:
        for t, ut in zip(times, predictions):
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
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


    res =  {"confusion_matrix": cm, "correlation":correlations.count(1), "rouge": scorer, "ref_timing": list(ref_timing),
            "pred_timing": list(pred_timing)}
    return res
