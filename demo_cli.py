from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from utils.modelutils import check_model_paths
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import argparse
import torch
import sys
import os
import nltk
from audioread.exceptions import NoBackendError
#gui
from train import wishMe,takeCommand,Query
import tkinter
from tkinter import filedialog
from tkinter import *
nltk.download('punkt')
if __name__ == '__main__':
	## Info & args
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument("-e", "--enc_model_fpath", type=Path, 
						default="encoder/saved_models/pretrained.pt",
						help="Path to a saved encoder")
	parser.add_argument("-s", "--syn_model_fpath", type=Path, 
						default="synthesizer/saved_models/pretrained/pretrained.pt",
						help="Path to a saved synthesizer")
	parser.add_argument("-v", "--voc_model_fpath", type=Path, 
						default="vocoder/saved_models/pretrained/pretrained.pt",
						help="Path to a saved vocoder")
	parser.add_argument("--cpu", action="store_true", help=\
		"If True, processing is done on CPU, even when a GPU is available.")
	parser.add_argument("--no_sound", action="store_true", help=\
		"If True, audio won't be played.")
	parser.add_argument("--seed", type=int, default=None, help=\
		"Optional random number seed value to make toolbox deterministic.")
	parser.add_argument("--no_mp3_support", action="store_true", help=\
		"If True, disallows loading mp3 files to prevent audioread errors when ffmpeg is not installed.")
	args = parser.parse_args()
	#print_args(args, parser)
	if not args.no_sound:
		import sounddevice as sd

	if args.cpu:
		# Hide GPUs from Pytorch to force CPU processing
		os.environ["CUDA_VISIBLE_DEVICES"] = ""

	if not args.no_mp3_support:
		try:
			librosa.load("samples/1320_00000.mp3")
		except NoBackendError:
			print("Librosa will be unable to open mp3 files if additional software is not installed.\n"
				  "Please install ffmpeg or add the '--no_mp3_support' option to proceed without support for mp3 files.")
			exit(-1)
		
	print("Running a test of your configuration...\n")
		
	if torch.cuda.is_available():
		device_id = torch.cuda.current_device()
		gpu_properties = torch.cuda.get_device_properties(device_id)
		## Print some environment information (for debugging purposes)
		print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
			"%.1fGb total memory.\n" % 
			(torch.cuda.device_count(),
			device_id,
			gpu_properties.name,
			gpu_properties.major,
			gpu_properties.minor,
			gpu_properties.total_memory / 1e9))
	else:
		print("Using CPU for inference.\n")
	
	## Remind the user to download pretrained models if needed
	check_model_paths(encoder_path=args.enc_model_fpath,
					  synthesizer_path=args.syn_model_fpath,
					  vocoder_path=args.voc_model_fpath)
	
	## Load the models one by one.
	print("Preparing the encoder, the synthesizer and the vocoder...")
	encoder.load_model(args.enc_model_fpath)
	synthesizer = Synthesizer(args.syn_model_fpath)
	vocoder.load_model(args.voc_model_fpath)
	
	
	## Run a test
	print("Testing your configuration with small inputs.")
	# Forward an audio waveform of zeroes that lasts 1 second. Notice how we can get the encoder's
	# sampling rate, which may differ.
	# If you're unfamiliar with digital audio, know that it is encoded as an array of floats 
	# (or sometimes integers, but mostly floats in this projects) ranging from -1 to 1.
	# The sampling rate is the number of values (samples) recorded per second, it is set to
	# 16000 for the encoder. Creating an array of length <sampling_rate> will always correspond 
	# to an audio of 1 second.
	print("\tTesting the encoder...")
	encoder.embed_utterance(np.zeros(encoder.sampling_rate))
	
	# Create a dummy embedding. You would normally use the embedding that encoder.embed_utterance
	# returns, but here we're going to make one ourselves just for the sake of showing that it's
	# possible.
	embed = np.random.rand(speaker_embedding_size)
	# Embeddings are L2-normalized (this isn't important here, but if you want to make your own 
	# embeddings it will be).
	embed /= np.linalg.norm(embed)
	# The synthesizer can handle multiple inputs with batching. Let's create another embedding to 
	# illustrate that
	embeds = [embed, np.zeros(speaker_embedding_size)]
	texts = ["test 1", "test 2"]
	print("\tTesting the synthesizer... (loading the model will output a lot of text)")
	mels = synthesizer.synthesize_spectrograms(texts, embeds)
	
	# The vocoder synthesizes one waveform at a time, but it's more efficient for long ones. We 
	# can concatenate the mel spectrograms to a single one.
	mel = np.concatenate(mels, axis=1)
	# The vocoder can take a callback function to display the generation. More on that later. For 
	# now we'll simply hide it like this:
	no_action = lambda *args: None
	print("\tTesting the vocoder...")
	# For the sake of making this test short, we'll pass a short target length. The target length 
	# is the length of the wav segments that are processed in parallel. E.g. for audio sampled 
	# at 16000 Hertz, a target length of 8000 means that the target audio will be cut in chunks of
	# 0.5 seconds which will all be generated together. The parameters here are absurdly short, and 
	# that has a detrimental effect on the quality of the audio. The default parameters are 
	# recommended in general.
	vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)


	def receive():
		global embed
		in_fpath = filedialog.askopenfilename()
		preprocessed_wav = encoder.preprocess_wav(in_fpath)
		# - If the wav is already loaded:
		original_wav,sampling_rate = librosa.load(in_fpath)
		preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
		print("Loaded file succesfully")        
		# Then we derive the embedding. There are many functions and parameters that the 
		# speaker encoder interfaces. These are mostly for in-depth research. You will typically
		# only use this function (with its default parameters):
		embed = encoder.embed_utterance(preprocessed_wav)

	def send():
		global embed
		global synthesizer
		q = takeCommand()
		ChatLog.insert(END, "User: " + q + '\n')
		qu = q.lower()
		text = Query(qu)
		ChatLog.insert(END, "Bot: " + text + '\n')
		if text==None:
			return
		print("Bot: "+text)
		if args.seed is not None:
			torch.manual_seed(args.seed)
			synthesizer = Synthesizer(args.syn_model_fpath)

		# The synthesizer works in batch, so you need to put your data in a list or numpy array
		texts = [text]
		embeds = [embed]
		# If you know what the attention layer alignments are, you can retrieve them here by
		# passing return_alignments=True
		specs = synthesizer.synthesize_spectrograms(texts, embeds)
		spec = specs[0]
		# good to see you

		
		
		## Generating the waveform
		print("Synthesizing the waveform:")

		# If seed is specified, reset torch seed and reload vocoder
		if args.seed is not None:
			torch.manual_seed(args.seed)
			vocoder.load_model(args.voc_model_fpath)

		# Synthesizing the waveform is fairly straightforward. Remember that the longer the
		# spectrogram, the more time-efficient the vocoder.
		generated_wav = vocoder.infer_waveform(spec)
		
		generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

		# Trim excess silences to compensate for gaps in spectrograms (issue #53)
		generated_wav = encoder.preprocess_wav(generated_wav)
		
		# Play the audio (non-blocking)
		if not args.no_sound:
			try:
				sd.stop()
				sd.play(generated_wav, synthesizer.sample_rate)
			except sd.PortAudioError as e:
				print("\nCaught exception: %s" % repr(e))
				print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
			except:
				raise
		global num_generated
		filename = "demo_output_%02d.wav" % num_generated
		print(generated_wav.dtype)
		sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
		num_generated += 1
		print("\nSaved output as %s\n\n" % filename)
		print("")

	   
  
	num_generated = 0
	base = Tk()
	base.title("Mini project ChatBot")
	base.geometry("400x500")
	base.resizable(width=FALSE, height=FALSE)


	ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

	#ChatLog.config(state=DISABLED)

	scrollbar = Scrollbar(base, command=ChatLog.yview)
	ChatLog['yscrollcommand'] = scrollbar.set
	SendButton = Button(base, font=("Verdana",12,'bold'), text="SPEAK NOW", width="12", height=5,
						bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
						command= send )
	TargetVoice = Button(base, font=("Verdana",12,'bold'), text="BROWSE TARGETVOICE", width="12", height=5,
						bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
						command= receive)


	scrollbar.place(x=376,y=6, height=386)
	ChatLog.place(x=6,y=6, height=386, width=370)
	SendButton.place(x=6, y=401, height=45, width=370)
	TargetVoice.place(x=6,y=447,height=45,width=370)
	wishMe()
	base.mainloop()

