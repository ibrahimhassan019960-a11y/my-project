import sys, subprocess, numpy as np
from faster_whisper import WhisperModel
m=WhisperModel(sys.argv[1], device='cpu', compute_type='int8')
a=np.frombuffer(subprocess.run(['ffmpeg','-nostdin','-v','error','-i',sys.argv[2],'-ac','1','-ar','16000','-f','s16le','-'],capture_output=True).stdout,np.int16).astype(np.float32)/32768
segs,_=m.transcribe(a, language='ar', beam_size=5, word_timestamps=True, condition_on_previous_text=False)
for s in segs:
    if s.end<float(sys.argv[3]) or s.start>float(sys.argv[4]): continue
    print(' '.join(f"{w.word.strip()}[{w.start:.2f}-{w.end:.2f}]" for w in s.words))
