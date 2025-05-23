
from openai import OpenAI
import pysrt
import sys
import os
import configparser
import shutil
import time
import concurrent.futures
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv("../.env")
# Initialize and read the configuration
config = configparser.ConfigParser()
config.read('config.ini')
global output_path
# Defining a unique marker
# Choose a unique sequence that won't appear in translations.
# (not in use in current version)
marker = " <|> "

# Constants for retry logic
MAX_RETRY_ATTEMPTS = 5
RETRY_INTERVAL = 10  # seconds

# print term width horizontal line
def print_horizontal_line(character='-'):
    terminal_width = shutil.get_terminal_size().columns
    line = character * terminal_width
    print(line, flush=True)

# Function to read the OpenAI API key securely
def get_api_key():
    prefer_env = config.getboolean('DEFAULT', 'PreferEnvForAPIKey', fallback=True)

    if prefer_env:
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key is not None:
            return api_key

    try:
        with open('api_token.txt', 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        if not prefer_env:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key is not None:
                return api_key

        print("The OPENAI_API_KEY environment variable is not set, and `api_token.txt` was not found.")
        print("Please set either one and adjust `config.ini` if needed for the preferred load order.")
        sys.exit(1)

# Read the config.ini / configuration
def get_config(section, option, prompt_message, is_int=False):
    if section not in config or option not in config[section]:
        print(prompt_message)
        value = input().strip()
        if not section in config:
            config.add_section(section)
        config.set(section, option, value)
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
        return int(value) if is_int else value
    return int(config[section][option]) if is_int else config[section][option]

# Function to split long lines without breaking words
def split_long_line(text, max_line_length):
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        if sum(len(w) + 1 for w in current_line) + len(word) <= max_line_length:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))
    return lines


# Function to translate blocks of subtitles with context-specific information
def translate_block(block, block_num, total_blocks, model, max_tokens,temperature):
    # print_horizontal_line()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    #print(f"::: [ Translating block {block_num} / {total_blocks} ]")
    original_indices = [sub.index for sub in block]
    combined_text = "\n".join([f"[{sub.index}] {sub.text}".replace('\n', ' ') for sub in block])

    # print("::: Input text:")
    # print_horizontal_line()
    # print(combined_text)

    translated_text = ""
    attempts = 0

    while attempts < MAX_RETRY_ATTEMPTS and not translated_text:
        try:
            prompt_text = (
                f"Translate this SRT file into English. This SRT file is a live commentary for a motor race in Japanese."
                f"Text to be translated: {combined_text}")

            chat_completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": prompt_text}],
                temperature=float(temperature),
                max_tokens=max_tokens
            )
            translated_text = chat_completion.choices[0].message.content.strip()

            if not translated_text:
                raise ValueError("Empty translation received")

        except Exception as e:
            print(f"Error during translation attempt {attempts + 1}: {e}")
            time.sleep(RETRY_INTERVAL)
            attempts += 1

    if not translated_text:
        print(f"::: Translation failed after {MAX_RETRY_ATTEMPTS} attempts.")
        translated_text = "\n".join([f"[{index}] Translation Error" for index in original_indices])

    # print_horizontal_line()
    # print("::: Translated text (verify indexes):")
    # print_horizontal_line()
    # print(translated_text)

    corrected_translations = []
    translated_lines = translated_text.split("\n")

    # Directly assign translated content without inserting index markers
    for i, text in enumerate(translated_lines):
        # Remove any numerical prefix mistakenly carried over from input
        cleaned_text = text.partition(']')[2].strip() if ']' in text else text.strip()
        if i < len(original_indices):
            corrected_translations.append(cleaned_text)
        else:
            # If there are more lines than expected, it breaks out or logs extra lines
            break

    # Handle any missing translation lines
    while len(corrected_translations) < len(original_indices):
        corrected_translations.append("<<MISSING TRANSLATION>>")

    return corrected_translations


def check_path_and_list_srt_files(path):
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        # List all .srt files in the directory
        srt_files = [os.path.join(path,file) for file in os.listdir(path) if file.endswith('.srt') and os.path.isfile(os.path.join(path, file))]
        return srt_files
    else:
        return f"The path '{path}' does not exist."



def translate(input_file_path):

    if not input_file_path.lower().endswith('.srt'):
        print("The provided file does not have an .srt extension.")
        sys.exit(1)

    # Load in the .srt file
    try:

        subs = pysrt.open(input_file_path)
    except Exception as e:
        print(f"Error reading the SRT file: {e}")
        sys.exit(1)

    # Determine the output file path based on the input
    output_file = os.path.basename(input_file_path).replace('.srt', '_translated.srt')
    output_path = sys.argv[2]
    output_file_path = os.path.join(output_path, output_file)

    try:
        # For integer values, explicitly specify is_int=True.
        block_size = get_config('Settings', 'BlockSize', "Please enter the number of subtitles to process at once (e.g., 10):", is_int=True)
        max_line_length = get_config('Settings', 'MaxLineLength', "Max characters per subtitle line:", is_int=True)
        # For string values again, no need to specify is_int.
        model = get_config('Settings', 'Model', "Please enter the model to use (e.g., 'gpt-3.5-turbo-0125'):")
        # Here is_int is not needed as temperature is not an integer.
        temperature = get_config('Settings', 'Temperature', "Please enter the temperature to use for translation (e.g., 0.3):")
        # Explicitly specify is_int=True for integer values.
        max_tokens = get_config('Settings', 'MaxTokens', "Please enter the max tokens to use for translation (e.g., 1024):", is_int=True)
    except Exception as e:
        print(f"Error retrieving configuration: {e}")
        sys.exit(1)
    # In the main translation loop:
    block_size = len(subs)
    total_blocks = (len(subs) + block_size - 1) // block_size

    # Translate block by block
    try:
        for i, start in enumerate(range(0, len(subs), block_size)):
            block = subs[start:start + block_size]
            translated_block = translate_block(block, i + 1, total_blocks, model, max_tokens,temperature)
            for j, sub in enumerate(block):
                if j < len(translated_block):
                    # Apply line splitting only if max_line_length is greater than zero.
                    if max_line_length > 0:
                        split_text = split_long_line(translated_block[j], max_line_length)
                        sub.text = '\n'.join(split_text)
                    else:
                        # If max_line_length is zero, use the text without splitting.
                        sub.text = translated_block[j]
                else:
                    sub.text = "Translation Error"
    except Exception as e:
        print(f"Error during translation process: {e}")
        sys.exit(1)

    # Save the translated subtitles
    subs.save(output_file_path)
    #print_horizontal_line()
    #print(f"::: Translation done!\n::: Translated subtitles saved to: {output_file_path}")
    #print_horizontal_line()
    return  output_file_path

if __name__ == '__main__':
    #freeze_support()
    if len(sys.argv) < 3:
        print("Usage: python translate_srt.py path/to/your/file/or/folder.srt")
        sys.exit(1)

    # Check the .srt file and load it in
    input_path = sys.argv[1]
    input_file_paths = check_path_and_list_srt_files(input_path)
    input_file_paths = input_file_paths
    # setup the output directory

    output_path = sys.argv[2]

    # If only a subset should be executed
    if len(sys.argv) > 3:
        start = int(sys.argv[3])
        end = int(sys.argv[4])
        if end > len(input_file_paths):
            end = len(input_file_paths)
    else:
        start = 0
        end = len(input_file_paths)


    input_file_paths = input_file_paths[start:end]

    if not os.path.exists(output_path):
        # If it does not exist, create the directory
        os.makedirs(output_path)

    # Start translation
    print(f"{len(input_file_paths)} files detected. \n Starting translation.")

    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for input_file_path in input_file_paths:
            futures.append(executor.submit(translate, input_file_path))
        for future in concurrent.futures.as_completed(futures):
            print(f"Translation done for file: {futures.index(future)}")

    end_time = time.time()
    print(f"Translated all provided files in {end_time - start_time:.2f} seconds")
