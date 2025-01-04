# --------------------------------------------------------------------------------------------
# This pipeline uses Nvidia NeMo, OpenAI Whisper, and Facebook Demucs to transcribe and diarize audio files
# Modified from https://github.com/MahmoudAshraf97/whisper-diarization
# Special thanks to Mahmoud Ashraf
# --------------------------------------------------------------------------------------------

import os
import time

# os.environ['OMP_NUM_THREADS'] = '16'


# options
input_files_path = '/uufs/chpc.utah.edu/common/HIPAA/IRB_00083132/studydata/'
output_files_path = '/uufs/chpc.utah.edu/common/HIPAA/IRB_00083132/new_diarized_transcription/'


# run the pipeline
# processed_files = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(output_files_path, 'txt'))]
with open('timelog.txt', 'r') as f:
    tmp = f.readlines()
processed_files = [f.split(':')[0] for f in tmp]
audio_files = [f for f in os.listdir(input_files_path) if f not in processed_files]
# audio_files = ['16035']

for audio_file in audio_files:

    input_path = os.path.join(input_files_path, f"{os.path.splitext(audio_file)[0]}")
    output_txt_path = os.path.join(output_files_path, 'txt', f"{os.path.splitext(audio_file)[0]}")
    output_srt_path = os.path.join(output_files_path, 'srt', f"{os.path.splitext(audio_file)[0]}")

    if not os.path.exists(os.path.join(output_files_path, 'txt')):
        os.makedirs(os.path.join(output_files_path, 'txt'), exist_ok=True)

    if not os.path.exists(os.path.join(output_files_path, 'srt')):
        os.makedirs(os.path.join(output_files_path, 'srt'), exist_ok=True)

    # apply pretrained pipeline
    start = time.time()
    os.system(f'python3 nemo+fasterwhisper.py --input_path {input_path} --output_paths {output_txt_path} {output_srt_path}')
    with open('timelog.txt', 'a') as f:
        f.write(f'{audio_file}: {round(time.time() - start, 6)}\n')


if __name__ == '__main__':
    pass
