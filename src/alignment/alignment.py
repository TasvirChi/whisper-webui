import argparse
import gc
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, List
import torch
import whisperx
from src.languages import get_language_names

class Alignment:
    def __init__(self):
        self.initialized = False
        self.pipeline = None

    def run(self, input_audio, result, **kwargs):

        language_code = kwargs.get('language_code', None)
        device = kwargs.get('device', "cuda" if torch.cuda.is_available() else "cpu")
        model_name = kwargs.get('model_name', None)
        model_dir = kwargs.get('model_dir', None)
        align_model, align_metadata = whisperx.load_align_model(language_code=language_code, device=device, model_name=model_name, model_dir=model_dir)
        
        # >> Align
        if align_model is not None and len(result["segments"]) > 0:
            if result.get("language", "en") != align_metadata["language"]:
                # load new language
                print(f"New language found ({result['language']})! Previous was ({align_metadata['language']}), loading new alignment model for new language...")
                align_model, align_metadata = whisperx.load_align_model(language_code=result["language"], device=device, model_name=model_name, model_dir=model_dir)
            print(">>Performing alignment...")
            interpolate_method = kwargs.get('interpolate_method', "nearest")
            return_char_alignments = kwargs.get('return_char_alignments', False)
            print_progress = kwargs.get('print_progress', False)
            combined_progress = kwargs.get('combined_progress', False)
            result = whisperx.align(transcript=result["segments"], model=align_model, align_model_metadata=align_metadata, 
                                    audio=input_audio, device=device, interpolate_method=interpolate_method, 
                                    return_char_alignments=return_char_alignments, print_progress=print_progress, 
                                    combined_progress=combined_progress)

        # Unload align model
        del align_model
        gc.collect()
        torch.cuda.empty_cache()
        return result


def main():
    from src.utils import write_srt
    from src.diarization.transcriptLoader import load_transcript

    parser = argparse.ArgumentParser(description='Add speakers to a SRT file or Whisper JSON file using pyannote/speaker-diarization.')
    parser.add_argument('audio_file', type=str, help='Input audio file')
    parser.add_argument('whisper_file', type=str, help='Input Whisper JSON/SRT file')
    # alignment params
    parser.add_argument("--language", type=str, default=None, choices=sorted(get_language_names()), \
                        help="language spoken in the audio, specify None to perform language detection")
    parser.add_argument("--align_model", default=None, help="Name of phoneme-level ASR model to do alignment")
    parser.add_argument("--interpolate_method", default="nearest", choices=["nearest", "linear", "ignore"], help="For word .srt, method to assign timestamps to non-aligned words, or merge them into neighbouring.")
    parser.add_argument("--char_alignments", action='store_true', help="Return character-level alignments in the output json file")


    args = parser.parse_args()

    print("\nReading whisper JSON from " + args.whisper_file)

    # Read whisper JSON or SRT file
    whisper_result = load_transcript(args.whisper_file)

    alignment = Alignment()
    alignment_result = list(alignment.run(args.audio_file, whisper_result, align_model=args.align_model, interpolate_method=args.interpolate_method, char_alignments=args.char_alignments))

    # Print result
    print("Alignment result:")
    for entry in alignment_result:
        print(f"  start={entry.start:.1f}s stop={entry.end:.1f}s speaker_{entry.speaker}")


if __name__ == "__main__":
    main()
    
    #test = Alignment()

    #input("Press Enter to continue...")