import argparse, os
from main import get_commentary_path
from pandas import DataFrame
from datasets import Dataset
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Generates commentary as per the defined settings"
    )
    parser.add_argument("--dir", required=True, type=str, help="Directory containing the videos "
                        "and respective commentary in recordings and transcriptions_whole_data_english folder")

    args = parser.parse_args()

    folder = args.dir
    video_directory = "recordings"
    video_directory = os.path.join(folder,video_directory)

    commentary_directory = "transcriptions_whole_data_english"
    commentary_directory = os.path.join(folder,commentary_directory)
    hf_dataset = []

    all_game_path = [os.path.join(video_directory, name) for name in os.listdir(video_directory) if
                     os.path.isdir(os.path.join(video_directory, name))]

    count = 0
    for i, game_path in enumerate(all_game_path):
        transcription_file = get_commentary_path(commentary_directory,game_path)
        if transcription_file is not None:
            mp4_file = [os.path.join(game_path,file) for file in os.listdir(game_path) if
                         file.endswith('.mp4') and os.path.isfile(os.path.join(game_path, file)) and "客観" in file][0]
        else:
            #print (f"kyakkan commentary not available for game: {game_path}")
            count += 1
            continue

        sample_name = os.path.dirname(mp4_file).split('/')[-1]
        dataset_item = {"sample_name": sample_name,
                        "video": mp4_file,
                        "srt": transcription_file, }
        hf_dataset.append(dataset_item)

    hf_dataset = Dataset.from_list(hf_dataset)
    dataset_processed = hf_dataset.shuffle(seed=42)
    print (f"kyakkan commentary not available for {count} samples.")
    print (dataset_processed)
    hf_dataset = dataset_processed.train_test_split(test_size=400)
    dir = "RaceCommentaryEn/"
    os.makedirs(dir, exist_ok=True)
    hf_dataset.save_to_disk(dir)
    #train_dataset, test_dataset = hf_dataset['train'].with_format("torch"), hf_dataset['test'].with_format("torch")





