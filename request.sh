port=$1

curl -v http://0.0.0.0:{$port}/v1/chat/completions \
	-H 'Content-Type: application/json' \
	-d \
	'{ "model": "/home/admin/model-csi/models/modelhub_35500009_deepseek-v3-2-w8a8-106300046_20260130104046/model",
	"messages": [
          {"role": "user", "content": "What is the capital of France?"}
	  ],
	  "temperature": 0.6,
	  "repetition_penalty": 1.0,
	  "top_p": 0.95,
	  "top_k": 40,
	  "max_tokens": 20,
	  "stream": false,
  	  "ignore_eos": false}' #\