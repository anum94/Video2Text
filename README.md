# video2Text

Usage

python main.py "path/to/data"

Example:
python main.py --dir "/groups/gac50547/RaceCommentary" --n 1 --icl True --step 20 --k 4 --frames 1 --context_window 4096 --wb False

python finetune.py --dir "/groups/gac50547/RaceCommentary/" --frames 2 --step 2 --n 2000


#transcription_file = "transcriptions_whole_data_english/AC_150221-130155_R_ks_porsche_macan_mugello__kyakkan.merged.mp4_translated.srt"
#transcription_file = os.path.join(folder, transcription_file)
#mp4_file = "AC_150221-130155_R_ks_porsche_macan_mugello_/AC_150221-130155_R_ks_porsche_macan_mugello_客観.mp4"
#mp4_file = os.path.join(video_directory, mp4_file)
