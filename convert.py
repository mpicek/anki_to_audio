#!/usr/bin/env python3
import gtts
from playsound import playsound

import csv
import os
import numpy as np

import pathlib
from tqdm import tqdm
from moviepy.editor import concatenate_audioclips, AudioFileClip
from pydub import AudioSegment
import pydub
import argparse
from pathlib import Path

from numpy import linspace,sin,pi,int16



def get_data_files(data_path):
    data_files = []
    for fn in os.listdir(data_path):
        data_files.append(Path(data_path) / Path(fn))
    return data_files

def concatenate_audio_moviepy(audio_clip_paths, output_path):
    """Concatenates several audio files into one audio file using MoviePy
    and save it to `output_path`. Note that extension (mp3, etc.) must be added to `output_path`"""
    clips = [AudioFileClip(c) for c in audio_clip_paths]
    final_clip = concatenate_audioclips(clips)
    final_clip.write_audiofile(output_path)

def generate_audio(args, data_files, format):
    for j, file_name in tqdm(enumerate(data_files)):
        deeper_output_dir = Path(args.output_path) / Path("single_words") / file_name.stem
        os.makedirs(deeper_output_dir, exist_ok=True)

        print(file_name)

        with open(file_name) as f:
            csv_reader = csv.reader(f, delimiter='\t')
            print("Processing file " + str(file_name))
            for r_number, row in tqdm(enumerate(csv_reader)):
                # data.append(row)

                for i in range(len(format) - 1):
                    new_path = deeper_output_dir / Path(f"{r_number:06d}_{i}_{format[i]}.mp3")
                    if not new_path.exists():
                        # print(new_path)
                        tts = gtts.gTTS(row[i], lang=format[i])
                        tts.save(new_path)

def concatenate_audia(audia):
    combined = AudioSegment.empty()
    for song in audia:
        combined += song
    
    return combined

def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y

def write(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")

# tone synthesis
def note(freq, len_in_ms, amp=1, rate=24000):
    t = linspace(0,len_in_ms,len_in_ms*(rate//1000))
    data = sin(2*pi*freq*t)*amp
    return data.astype(int16) # two byte integers

def pause(len_in_ms, rate=24000):
    return np.zeros((int(len_in_ms*(rate//1000)),))

def generate_final_audio(args, data_files, format):
    next_card_tone = note(440,300,amp=10000)
    next_card_pause_audio = pause(args.next_card_pause)
    thinking_pause_audio = pause(args.thinking_pause)

    for j, file_name in tqdm(enumerate(data_files)):
        deeper_output_dir = Path(args.output_path) / Path("single_words") / file_name.stem

        with open(file_name) as f:
            csv_reader = csv.reader(f, delimiter='\t')
            # print("Processing file " + str(file_name))
            audio_flashcards = []
            for r_number, row in enumerate(csv_reader):
                # data.append(row)


                # for i in range(len(format) - 1):
                first_side_mp3 = deeper_output_dir / Path(f"{r_number:06d}_{args.first_side}_{format[args.first_side]}.mp3")
                second_side_mp3 = deeper_output_dir / Path(f"{r_number:06d}_{1 - args.first_side}_{format[1 - args.first_side]}.mp3")

                example_mp3 = None
                if args.include_example:
                    example_mp3 = deeper_output_dir / Path(f"{r_number:06d}_{2}_{format[2]}.mp3")
                    # if not new_path.exists():
                        # print(new_path)
                
                sr1, x1 = read(first_side_mp3)
                sr2, x2 = read(second_side_mp3)
                sr_ex, x_ex = None, None

                if args.include_example:
                    sr_ex, x_ex = read(first_side_mp3)

                audio_flashcards.append(x1)
                audio_flashcards.append(thinking_pause_audio)
                audio_flashcards.append(x2)
                audio_flashcards.append(next_card_pause_audio)
                audio_flashcards.append(next_card_tone)
                audio_flashcards.append(next_card_pause_audio)

                if (r_number + 1) % args.cards_per_mp3 == 0:
                    flashcards_output_dir = Path(args.output_path) / Path("flashcards")
                    os.makedirs(flashcards_output_dir, exist_ok=True)
                    out = np.concatenate(audio_flashcards)
                    write(flashcards_output_dir / Path(f""))
                    audio_flashcards = []


            x = np.concatenate((p, tone, p))

            write('out2.mp3', sr, x)
            playsound('out2.mp3')
                



def main(args):
    
    data_files = get_data_files(args.data_path)
    print(data_files)
    format = ['en', 'cs', 'en']
    

    generate_audio(args, data_files, format)
    
    # playsound("output/" + "02.mp3")


    # audia = []
    # for audio in os.listdir(args.output_path / Path("single_words")):
    #     a = AudioSegment.from_mp3("output/" + audio)
    #     audia.append(a)

    # combined = concatenate_audia(audia)
    # combined.export("output/output.mp3", format="mp3")


    # def write(f, sr, x, normalized=False):
    #     """numpy array to MP3"""
    #     channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    #     if normalized:  # normalized array - each item should be a float in [-1, 1)
    #         y = np.int16(x * 2 ** 15)
    #     else:
    #         y = np.int16(x)
    #     song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    #     song.export(f, format="mp3", bitrate="320k")

    #[[-225  707]
    # [-234  782]
    # [-205  755]
    # ..., 
    # [ 303   89]
    # [ 337   69]
    # [ 274   89]]

    



parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_path", default="data", type=str, help="Folder with anki input data.")
parser.add_argument("--output_path", default="output", type=str, help="Output folder.")
parser.add_argument("--thinking_pause", default=2000, type=int, help="Pause for thinking in milliseconds.")
parser.add_argument("--next_card_pause", default=200, type=int, help="Pause for the next card.")
parser.add_argument("--cards_per_mp3", default=100, type=int, help="Pause for the next card.")
parser.add_argument("--first_side", default=1, type=int, help="Index to format array of the first card.")
parser.add_argument("--include_example", default=False, action="store_true", help="Pause for the next card.")
# parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
