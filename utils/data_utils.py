import pysrt
import sys

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

