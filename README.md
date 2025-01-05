# Automatic Transcription and Diarization of Audio Recordings
This project is used to automatically process audio recordings of two-person dialogues, which can be useful for 
therapy session transcription. The pipeline uses Nvidia NeMo, OpenAI Whisper, and Facebook Demucs to transcribe and 
diarize audio files.  
**NOTE**: The model has not been fine-tuned with the therapy session data.  

Special thanks to Mahmoud Ashraf. The current project is based on his work.

## Installation
1. Clone the repository
```bash
git clone https://github.com/XinY-Z/diarization-transcription.git
```
2. Install the required packages
```bash
pip install -r requirements.txt
```

## Usage
1. Make sure cudnn and cublas are in the dynamic library path before opening Python
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<your cudnn path>/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<your cublas path>/lib
```
2. Put all the audio files you want to transcribe in a folder.
3. Open `run.py` and customize your file paths. `output_files_path` will be automatically created if not exist.
```python
input_files_path = <your input files path>  # The folder containing the audio files
output_files_path = <your output files path>  # The folder to save the diarized transcription
```
4. Run the script
```bash
python3 run.py
```

## Results
Two subfolders will be created in the `output_files_path`:
1. `txt`: Contains the diarized transcriptions in talk turns.
2. `srt`: Contains the diarized transcription in sentences, including the start and end time of each sentence.

## Example results
### Diarized transcription in talk turns
Transcripts are anonymized for confidentiality reasons. Only part of the transcript is shown here.  
Note that because the model has not been fine-tuned with therapy sessions, users may need to manually assign the speaking
roles to the speakers (i.e., client vs. therapist).
```txt
Speaker 1: Okay, so we'll just do a massage.  Have a seat wherever you feel most comfortable.  So how are you feeling? 
 What brings you into the counseling center?  

Speaker 0: I'm just feeling... I had my little intake appointment kind of explained like I'm just not feeling like myself 
and that's like starting to affect my schoolwork kind of losing that.  like motivation and determination that I had that 
like passion for like the field that I'm in and like pharmacy school in general I'm having a hard time kind of figuring out 
I don't know.  I like I've always had like been like a really motivated and determined person and lately I'm just like it's 
hard to just do like regular things let alone find the time to focus on school and...[truncated]
```

### Diarized transcription in sentences
Transcripts are anonymized for confidentiality reasons. Only part of the transcript is shown here.
```srt
1
00:00:45,020 --> 00:00:45,880
Speaker 1: Okay, so we'll just do a massage.

2
00:00:46,140 --> 00:00:48,820
Speaker 1: Have a seat wherever you feel most comfortable.

3
00:00:53,840 --> 00:01:04,280
Speaker 1: So how are you feeling?

4
00:01:04,980 --> 00:01:08,360
Speaker 1: What brings you into the counseling center?

5
00:01:08,400 --> 00:01:25,240
Speaker 0: I'm just feeling... I had my little intake appointment kind of explained like I'm just not feeling like myself 
and that's like starting to affect my schoolwork kind of losing that.

6
00:01:25,320 --> 00:01:43,880
Speaker 0: like motivation and determination that I had that like passion for like the field that I'm in and like pharmacy 
school in general I'm having a hard time kind of figuring out I don't know.

7
00:01:44,480 --> 00:02:08,479
Speaker 0: I like I've always had like been like a really motivated and determined person and lately I'm just like it's 
hard to just do like regular things let alone find the time to focus on school and...[truncated]
```

## Acknowledgements
Ashraf, M. (2024). _Whisper diarization: Speaker diarization using OpenAI Whisper_ [GitHub repository]. 
https://github.com/MahmoudAshraf97/whisper-diarization
