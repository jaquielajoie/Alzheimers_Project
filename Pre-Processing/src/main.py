import os
import sys
import argparse
import time
from audio_processing.AudioProcessor import AudioProcessor
from audio_processing.Modeler import Modeler
from audio_processing.Predictor import Predictor

"""
https://cgu.zoom.us/j/86581448382
"""

def set_cmd_args(args):
    if args.print:
        cmd_print = args.print
    else:
        cmd_print = False

    if args.max_file_count:
        max_file_count = args.max_file_count
    else:
        max_file_count = None

    if args.process_large_files:
        process_large_files = args.process_large_files
    else:
        process_large_files = False

    if args.analyze_mp3s:
        analyze_mp3s = args.analyze_mp3s
    else:
        analyze_mp3s = False

    if args.play_audio:
        play_audio = args.play_audio
    else:
        play_audio = False

    if args.audio_length:
        audio_length = args.audio_length
    else:
        audio_length = 10 #seconds

    if args.build_model:
        build_model = args.build_model
    else:
        build_model = False

    if args.predict_outcome:
        predict_outcome = args.predict_outcome
    else:
        predict_outcome = False

    return cmd_print, max_file_count, process_large_files, analyze_mp3s, play_audio, audio_length, build_model, predict_outcome

if __name__ == "__main__":

    #CMD Args settings
    parser = argparse.ArgumentParser(description='Process audio files.')
    parser.add_argument("-p", "--print", type=bool,
                    help="Print pyplot graphs.")
    parser.add_argument("-mf", "--max_file_count", type=int,
                    help="Maximum files to analyze.")
    parser.add_argument("-plf", "--process_large_files", type=bool,
                    help="Transform PCM/Wav files to MP3.")
    parser.add_argument("-a", "--analyze_mp3s", type=bool,
                    help="Analyze MP3s.")
    parser.add_argument("-pa", "--play_audio", type=bool,
                    help="Play audio along with pyplots.")
    parser.add_argument("-al", "--audio_length", type=int,
                    help="Time in seconds to play audio.")
    parser.add_argument("-m", "--build_model", type=bool,
                    help="Build a model from /json/data.json")
    parser.add_argument("-pred", "--predict_outcome", type=bool,
                    help="Predict all samples with first model")
    args = parser.parse_args()
    #print(args.echo)

    cmd_print, max_file_count, process_large_files, analyze_mp3s, play_audio, audio_length, build_model, predict_outcome = set_cmd_args(args)

    DATA_PATH = os.path.abspath("../test_data")
    processor = AudioProcessor(data_path=DATA_PATH)

    start_time = time.perf_counter()
    processor.run(cmd_print=cmd_print, max_file_count=max_file_count,
        process_large_files=process_large_files, analyze_mp3s=analyze_mp3s,
        play_audio=play_audio, audio_length=audio_length)

    end_time = time.perf_counter()

    print(f"""\n\nAudioProcessor.run(): took {end_time - start_time} seconds to run.\n
            process_large_files: {process_large_files}
            analyze_mp3s: {analyze_mp3s}
            max_file_count: {max_file_count}
            graphing: {cmd_print}
            play_audio: {play_audio}
            """)

    if build_model:
        print(f'Starting the model build from ./audio_processing/json/data.json ...')

        JSON_PATH = os.path.abspath("audio_processing/json/data.json")
        modeler =  Modeler(data_path=JSON_PATH)

        start_time = time.perf_counter()
        modeler.build()

        end_time = time.perf_counter()

        print(f"""\n\nModeler.build(): took {end_time - start_time} seconds to run.\n
                """)

    if predict_outcome:
        MODEL_PATH = os.path.abspath("../models/classifier.h5")
        PREDICT_FILE_PATH = os.path.abspath("../unlabeled_data/predict_1.mp3")
        predictor = Predictor(model_path=MODEL_PATH, file_path=PREDICT_FILE_PATH)

        start_time = time.perf_counter()
        predictor.predict()

        end_time = time.perf_counter()

        print(f"""\n\Predictor.predict(): took {end_time - start_time} seconds to run.\n
                """)
