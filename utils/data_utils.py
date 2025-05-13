import numpy as np
import pysrt
import os
import nltk
from sklearn.metrics import confusion_matrix
import json
from datetime import datetime
from scipy.stats import pearsonr
from bert_score import BERTScorer
import pysrt
from difflib import SequenceMatcher
from rouge_score import rouge_scorer
def get_text_sequence(lines):
    return [l['text'] for l in lines if l['text']]

def compute_LA(ref_lines, hyp_lines):
    ref_seq = get_text_sequence(ref_lines)
    hyp_seq = get_text_sequence(hyp_lines)
    # SequenceMatcher finds the longest contiguous matching subsequence
    sm = SequenceMatcher(None, ref_seq, hyp_seq)
    blocks = sm.get_matching_blocks()
    # Length of the longest matching block is the LA (can sum lengths if you want total, or count contiguous only)
    la = max(block.size for block in blocks)
    return la, blocks

def overlap(start1, end1, start2, end2):
    return max(start1, start2) < min(end1, end2)


def parse_srt(srt_path):
    subs = pysrt.open(srt_path)
    lines = []
    for item in subs:
        lines.append({
            'start': item.start.ordinal,  # in milliseconds
            'end': item.end.ordinal,
            'text': item.text.strip()
        })
    return lines
def compute_LAAL(ref_lines, hyp_lines, text_similarity_threshold=0.5):
    import difflib
    aligned = 0
    for ref in ref_lines:
        best_match = None
        for hyp in hyp_lines:
            if overlap(ref['start'], ref['end'], hyp['start'], hyp['end']):
                sm = difflib.SequenceMatcher(None, ref['text'], hyp['text'])
                sim = sm.ratio()
                if sim > text_similarity_threshold:
                    best_match = sim
        if best_match:
            aligned += 1
    # LAAL as a ratio or count (e.g., aligned actions / total reference actions)
    laal = aligned / len(ref_lines) if ref_lines else 0
    return laal

def srt_time_to_seconds(srt_time, ms = False):
    # Split the time string by colon and comma
    try:
        hours, minutes, seconds, milliseconds  = srt_time.hours, srt_time.minutes, srt_time.seconds, srt_time.milliseconds
        # Convert each part to an integer
        hours = int(hours)
        minutes = int(minutes)
        seconds = int(seconds)
        if ms:
            milliseconds = float(milliseconds)
        else:
            milliseconds = 0

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
        return (-1)

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

def write_logs(out_folder, predictions,times, mode, talking_speed_sample=None):
    out_file = os.path.join(out_folder, f"logs_{mode}.txt")
    #print(f"Generation stored at {out_file}")
    with open(out_file, 'a') as the_file:
        for t, ut in zip(times, predictions):
            ut = ut.replace("\n", "")
            the_file.write(f"{t}: {ut}\n")

    out_file = convert_text_to_srt(out_file, talking_speed_sample)
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


def flatten_2d_dict(in_dict:dict)->dict:
  out_dict = dict()
  print (in_dict)
  for key, value in in_dict.items():
    for v, k  in zip(value, ["p", "r", "f1"]):
      out_dict[f"{key}_{k}"] = v

  return out_dict


def interval_indices(length, n_intervals=10):
    """Helper to get list of (start, end) indices for `n_intervals` intervals."""
    indices = []
    for i in range(n_intervals):
        start_idx = int(i * length / n_intervals)
        end_idx = int((i + 1) * length / n_intervals)
        indices.append((start_idx, end_idx))
    return indices


def compute_10_percent(ref_list, pred_list, n_intervals = 10):
    assert len(pred_list) == len(ref_list), "Lists must be of the same length"


    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    bertscore = BERTScorer(model_type='bert-base-uncased')
    intervals = interval_indices(len(pred_list), n_intervals)
    score_dict = {}
    for i, (start, end) in enumerate(intervals):
        hyp = " ".join(pred_list[start:end])
        ref = " ".join(ref_list[start:end])
        rouge_score = rouge.score(ref, hyp)
        _, _, bert_F1 = bertscore.score([hyp], [ref])
        BLEUscore = nltk.translate.bleu_score.sentence_bleu([ref], [hyp], weights=(0.5, 0.5))


        score_dict[ f"rouge_{i * 10}-{(i + 1) * 10}%"] = rouge_score['rougeL'].fmeasure
        score_dict[f"bertscore_{i * 10}-{(i + 1) * 10}%"] = bert_F1
        score_dict[f"bleu_{i * 10}-{(i + 1) * 10}%"] = BLEUscore

    return score_dict


def compute_metrics(ref_timing, pred_timing, pred_utterences, ref_utterences, generated_srt, reference_srt):
    #print (len(ref_timing), len(pred_timing))
    correlations = [1 if a == b else 0 for a, b in zip(ref_timing, pred_timing)]

    p_corr, _ = pearsonr([1 if i==True else 0 for i in ref_timing],
                         [1 if i==True else 0 for i in pred_timing])
    cm = confusion_matrix(ref_timing, pred_timing)

    metrics_over_intervals = compute_10_percent(ref_utterences, pred_utterences)


    pred_commentary = "\n".join(pred_utterences)
    ref_commentary = "\n".join(ref_utterences)
    r_scorer = rouge_scorer.RougeScorer([ 'rougeL'], use_stemmer=True)
    rouge = r_scorer.score(ref_commentary, pred_commentary)
    rouge_L = rouge['rougeL'].fmeasure


    # BERTScore calculation
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, bert_F1 = scorer.score([pred_commentary], [ref_commentary])
    #print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")


    #flatten_2d_dict(rouge)


    BLEUscore = nltk.translate.bleu_score.sentence_bleu([ref_commentary], pred_commentary, weights=(0.5, 0.5))


    ref_lines = parse_srt(reference_srt)
    hyp_lines = parse_srt(generated_srt)

    # LA: longest contiguous block
    la, _ = compute_LA(ref_lines, hyp_lines)
    #print("Longest Alignment (LA):", la)


    # LAAL: fraction of reference actions aligned
    laal = compute_LAAL(ref_lines, hyp_lines)
    #print("Longest Aligned Action Location (LAAL):", laal)


    res =  {"correlation":(correlations.count(1))/len(correlations), "ROUGE_L": rouge_L,
            "BLEU": BLEUscore,  "ref_timing": list(ref_timing), "pearson": p_corr,
            "pred_timing": list(pred_timing), "bins": metrics_over_intervals, "BERTScore": bert_F1,
            'LAAL': laal, "LA": la}
    return res


def estimate_talking_speed(sample_file):
    sample_commentary = read_srt(sample_file)
    words_per_second = []
    for utterence in sample_commentary:
        time_to_speak = srt_time_to_seconds(utterence.duration, ms = True)
        num_of_words = len(utterence.text.split())
        if time_to_speak != 0:
            words_per_second.append(float(num_of_words/time_to_speak))

    return np.mean(words_per_second)

def convert_text_to_srt(file_path: str = None, talking_speed_sample:str = "../RaceCommentary/transcriptions_whole_data_english/AC_120221-180622_R_ks_audi_r8_plus_ks_nurburgring_layout_sprint_a_kyakkan.merged.mp4_translated.srt" ):
    if file_path is None:
        print("r Filepath with timestamps should be provided")
        exit()
    # seconds_per_word = estimate_talking_speed(
    #     sample_file=talking_speed_sample)
    seconds_per_word = 1 / estimate_talking_speed(sample_file=talking_speed_sample)
    if file_path is not None:
        timestamps = []
        utterances = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
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

        test = read_srt(srt_filename)
        #print(f"SRT file '{srt_filename}' created successfully.")
        return srt_filename
def rename_mp4_files(directory):
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.mp4'):
                # Customize your new name pattern as needed
                new_name = f"{filename.replace('%E5%AE%A2%E8%A6%B3', '客観')}"

                # Construct full file paths
                old_file = os.path.join(dirpath, filename)
                new_file = os.path.join(dirpath, new_name)

                # Rename the file
                os.rename(old_file, new_file)
                print(f'Renamed: {old_file} to {new_file}')

#dir = "../RaceCommentary/recordings/"
#rename_mp4_files(dir)