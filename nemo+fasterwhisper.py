# --------------------------------------------------------------------------------------------
# This pipeline uses Nvidia NeMo, OpenAI Whisper, and Facebook Demucs to transcribe and diarize audio files
# Modified from https://github.com/MahmoudAshraf97/whisper-diarization
# Special thanks to Mahmoud Ashraf
# --------------------------------------------------------------------------------------------

import torch
import torchaudio
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
import re
import faster_whisper
# import time
import argparse
import gc
from ctc_forced_aligner import (
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)
from helper_functions import *


## Options
parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default=None)
parser.add_argument("--output_paths", type=str, nargs=2, default=None)
args = parser.parse_args()

audio_path = args.input_path  # Name of the audio file
output_paths = args.output_paths  # Names of the output files [0] = txt, [1] = srt
# Whether to enable background/music removal from speech, helps increase diarization quality but uses alot of ram
enable_stemming = True
# (choose from 'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1',
# 'large-v2', 'large-v3', 'large')
whisper_model_name = "large-v3"
# replaces numerical digits with their pronounciation, increases diarization accuracy
suppress_numerals = True
batch_size = 10  # TODO: modify batch size if using smaller gpus
language = 'en'  # detect english only
device = "cuda" if torch.cuda.is_available() else "cpu"
device_index = 0
num_workers = 1


## Processing
## Separating music from speech using Demucs
# By isolating the vocals from the rest of the audio, it becomes easier to identify and track individual speakers based
# on the spectral and temporal characteristics of their speech signals. Source separation is just one of many techniques
# that can be used as a preprocessing step to help improve the accuracy and reliability of overall diarization process.

# start = time.time()
if enable_stemming:
    # Isolate vocals from the rest of the audio

    return_code = os.system(
        f'python -m demucs.separate -n htdemucs --two-stems=vocals "{audio_path}" -o "temp_outputs" --device "{device}"'
    )

    if return_code != 0:
        logging.warning("Source splitting failed, using original audio file.")
        vocal_target = audio_path
    else:
        vocal_target = os.path.join(
            "temp_outputs",
            "htdemucs",
            os.path.splitext(os.path.basename(audio_path))[0],
            "vocals.wav",
        )
else:
    vocal_target = audio_path
# print(f"Source separation took {time.time() - start:.2f}s")     # 65s (rtx5000)


## Transcriping audio using Whisper and realligning timestamps using Forced Alignment
# This code uses two different open-source models to transcribe speech and perform forced alignment on the resulting
# transcription. The first model is called OpenAI Whisper, which is a speech recognition model that can transcribe
# speech with high accuracy. The code loads the whisper model and uses it to transcribe the vocal_target file.
# The output of the transcription process is a set of text segments with corresponding timestamps indicating when each
# segment was spoken.

compute_type = "float16"
# or run on GPU with INT8
# compute_type = "int8_float16"
# TODO: to run on 1080ti, use float32
# compute_type = "float32"

whisper_model = faster_whisper.WhisperModel(
    whisper_model_name,
    device=device,
    device_index=device_index,
    num_workers=num_workers,
    compute_type=compute_type
)
whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
audio_waveform = faster_whisper.decode_audio(vocal_target)
suppress_tokens = (
    find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
    if suppress_numerals
    else [-1]
)

if batch_size > 0:
    transcript_segments, info = whisper_pipeline.transcribe(
        audio_waveform,
        language,
        suppress_tokens=suppress_tokens,
        batch_size=batch_size,
        without_timestamps=True,
    )
else:
    transcript_segments, info = whisper_model.transcribe(
        audio_waveform,
        language,
        suppress_tokens=suppress_tokens,
        without_timestamps=True,
        vad_filter=True,
    )

# start = time.time()
full_transcript = "".join(segment.text for segment in transcript_segments)
# print(f"Transcription took {time.time() - start:.2f}s")  # 364s (1080ti) 114s (rtx5000)

# clear gpu vram
del whisper_model, whisper_pipeline
torch.cuda.empty_cache()


## Aligning the transcription with the original audio using Forced Alignment.
# Forced alignment aims to align the transcription segments with the original audio signal contained in the
# vocal_target file. This process involves finding the exact timestamps in the audio signal where each segment was
# spoken and aligning the text accordingly.
# By combining the outputs of the two models, the code produces a fully aligned transcription of the speech contained
# in the vocal_target file. This aligned transcription can be useful for a variety of speech processing tasks, such as
# speaker diarization, sentiment analysis, and language identification.

alignment_model, alignment_tokenizer = load_alignment_model(
    device,
    # attn_implementation="sdpa",
    dtype=torch.float16 if device == "cuda" else torch.float32,
)  # TODO: need to switch to sdpa if using 1080ti

audio_waveform = (
    torch.from_numpy(audio_waveform)
        .to(alignment_model.dtype)
        .to(alignment_model.device)
)

emissions, stride = generate_emissions(
    alignment_model, audio_waveform, batch_size=batch_size
)

del alignment_model
torch.cuda.empty_cache()

tokens_starred, text_starred = preprocess_text(
    full_transcript,
    romanize=True,
    language=langs_to_iso[info.language],
)

# start = time.time()
segments, scores, blank_token = get_alignments(
    emissions,
    tokens_starred,
    alignment_tokenizer,
)
# print(f"Alignment took {time.time() - start:.2f}s")     # 42.2s (rtx5000)

spans = get_spans(tokens_starred, segments, blank_token)

word_timestamps = postprocess_results(text_starred, spans, stride, scores)


## Convert audio to mono for NeMo combatibility
ROOT = os.getcwd()
temp_path = os.path.join(ROOT, "temp_outputs")
if os.path.exists(temp_path):
    cleanup(temp_path)
os.makedirs(temp_path, exist_ok=True)
torchaudio.save(
    os.path.join(temp_path, "mono_file.wav"),
    audio_waveform.cpu().unsqueeze(0).float(),
    16000,
    channels_first=True,
)


## Speaker Diarization using NeMo MSDD Model
# This code uses a model called Nvidia NeMo MSDD (Multi-scale Diarization Decoder) to perform speaker diarization on
# an audio signal. Speaker diarization is the process of separating an audio signal into different segments based on
# who is speaking at any given time.

# Initialize NeMo MSDD diarization model
msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to("cuda")

# start = time.time()
msdd_model.diarize()  # takes time
# print(f"Speaker diarization took {time.time() - start:.2f}s")   # 39s (rtx5000)

del msdd_model
torch.cuda.empty_cache()

## Mapping Spekers to Sentences According to Timestamps
# Reading timestamps <> Speaker Labels mapping

speaker_ts = []
with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(" ")
        s = int(float(line_list[5]) * 1000)
        e = s + int(float(line_list[8]) * 1000)
        speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")


## Realligning Speech segments using Punctuation
# This code provides a method for disambiguating speaker labels in cases where a sentence is split between two different
# speakers. It uses punctuation markings to determine the dominant speaker for each sentence in the transcription.
#
# ```
# Speaker A: It's got to come from somewhere else. Yeah, that one's also fun because you know the lows are
# Speaker B: going to suck, right? So it's actually it hits you on both sides.
# ```
#
# For example, if a sentence is split between two speakers, the code takes the mode of speaker labels for each word in
# the sentence, and uses that speaker label for the whole sentence. This can help to improve the accuracy of speaker
# diarization, especially in cases where the Whisper model may not take fine utterances like "hmm" and "yeah" into
# account, but the Diarization Model (Nemo) may include them, leading to inconsistent results.

# The code also handles cases where one speaker is giving a monologue while other speakers are making occasional
# comments in the background. It ignores the comments and assigns the entire monologue to the speaker who is speaking
# the majority of the time. This provides a robust and reliable method for realigning speech segments to their
# respective speakers based on punctuation in the transcription.

if info.language in punct_model_langs:
    # restoring punctuation in the transcript to help realign the sentences
    punct_model = PunctuationModel(model="kredor/punctuate-all")

    words_list = list(map(lambda x: x["word"], wsm))

    labled_words = punct_model.predict(words_list, chunk_size=230)

    ending_puncts = ".?!"
    model_puncts = ".,;:!?"

    # We don't want to punctuate U.S.A. with a period. Right?
    is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

    for word_dict, labeled_tuple in zip(wsm, labled_words):
        word = word_dict["word"]
        if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
        ):
            word += labeled_tuple[1]
            if word.endswith(".."):
                word = word.rstrip(".")
            word_dict["word"] = word

else:
    logging.warning(
        f"Punctuation restoration is not available for {info.language} language. Using the original punctuation."
    )

wsm = get_realigned_ws_mapping_with_punctuation(wsm)
ssm = get_sentences_speaker_mapping(wsm, speaker_ts)


## Cleanup and Exporting the results

with open(f"{output_paths[0]}.txt", "w", encoding="utf-8-sig") as f:
    get_speaker_aware_transcript(ssm, f)

with open(f"{output_paths[1]}.srt", "w", encoding="utf-8-sig") as srt:
    write_srt(ssm, srt)

cleanup(temp_path)
del alignment_tokenizer, audio_waveform, emissions, punct_model, scores
torch.cuda.empty_cache()
gc.collect()


if __name__ == "__main__":
    pass
