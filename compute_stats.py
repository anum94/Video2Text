import argparse
import datetime
import os
from janome.tokenizer import Tokenizer
import numpy as np
import datasets
from tqdm import tqdm
from utils.video_utils import get_video_info
from utils.data_utils import read_srt
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Prepares samples for Human Evaluation from previously conducted experiments."
    )

    parser.add_argument("--hf_dataset", required=True, type=str, help="directoty with all the videos")

    args = parser.parse_args()

    hf_dataset = args.hf_dataset
    ds = datasets.load_from_disk(hf_dataset)
    test_dataset = ds['test'].with_format("torch")
    length = []
    srts = []

    for i in tqdm(range(len(test_dataset))):
        # get sample
        mp4_file = test_dataset[i]["video_path"]
        transcription_file = test_dataset[i]["srt_path"]

        metadata = get_video_info(mp4_file)
        length.append(metadata["duration"])
        srt = read_srt(transcription_file)
        if "Ja" in hf_dataset:
            tokenizer = Tokenizer()
            tokens = list(tokenizer.tokenize(srt.text))
            srt_count = [token.surface for token in tokens if
                     token.surface.strip() and token.part_of_speech.split(',')[0] != '記号']
        else:
            srt_count = len(srt.text.split())
        if srt_count > 0:
            srts.append(srt_count)

    maximum = max(length)
    minimum = min(length)
    average = sum(length) / len(length)



    print("Average Video Duration:", average)
    print("Minimum Video Duration:", minimum)
    print("Maximum Video Duration:", maximum)

    maximum = max(srts)
    minimum = min(srts)
    average = sum(srts) / len(srts)



    print("Average Commentary words:", average)
    print("Minimum Commentary words:", minimum)
    print("Maximum Commentary words:", maximum)









