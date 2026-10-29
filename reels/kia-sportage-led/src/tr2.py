import sys, subprocess, numpy as np
from faster_whisper import WhisperModel
m=WhisperModel(sys.argv[1], device='cpu', compute_type='int8')
for f in sys.argv[2:]:
    print('==',f.split('/')[-1])
    a=np.frombuffer(subprocess.run(['ffmpeg','-nostdin','-v','error','-i',f,'-ac','1','-ar','16000','-f','s16le','-'],capture_output=True).stdout,np.int16).astype(np.float32)/32768
    segs,_=m.transcribe(a, language='ar', beam_size=5, vad_filter=False, word_timestamps=True, condition_on_previous_text=False)
    for s in segs:
        print(f"{s.start:6.2f}-{s.end:6.2f} {s.text}")
