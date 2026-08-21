tp ?= 1
dp ?= 8
max_new_tokens ?= 2048
batch_size ?= 1024
temperature ?= 0.0
limit ?= -1
overwrite ?= False
model_path ?= Qwen/Qwen2.5-7B-Instruct
tokenizer_path ?= null
exp_name ?= Qwen2.5-7B-Instruct
output_dir ?= outputs/eval_llm/ 
suffix_prompt ?= Return your final response within \\boxed{{}}.
force_think ?= False
keep_think_below_budget_times ?= 0
timeout ?= 1800
seed ?= 42

all: eval_llm

eval_llm: eval_llm-start_server eval_llm-only_inference stop_server

eval_llm-start_server:
	@echo "Evaluating LLM baselines"
	@echo "model_path=${model_path}"
	@echo "tp=${tp}"
	@echo "dp=${dp}"

	python src/eval/inference_keep_think.py -u "model_path=${model_path},tp=${tp},dp=${dp}" --only_start_server

eval_llm-only_inference:
	@echo "model_path=${model_path}"
	@echo "max_new_tokens=${max_new_tokens}"
	@echo "batch_size=${batch_size}"
	@echo "suffix_prompt=${suffix_prompt}"
	@echo "temperature=${temperature}"
	@echo "limit=${limit}"
	@echo "overwrite=${overwrite}"
	@echo "output_dir=${output_dir}"
	@echo "tokenizer_path=${tokenizer_path}"
	@echo "force_think=${force_think}"
	@echo "keep_think_below_budget_times=${keep_think_below_budget_times}"
	@echo "timeout=${timeout}"
	@echo "seed=${seed}"

	python src/eval/inference_keep_think.py -u \
	"\
	suffix_prompt=${suffix_prompt},\
	output_dir=${output_dir},\
	model_path=${model_path},\
	exp_name=${exp_name},\
	max_new_tokens=${max_new_tokens},\
	batch_size=${batch_size},\
	temperature=${temperature},\
	limit=${limit},\
	overwrite=${overwrite},\
	tokenizer_path=${tokenizer_path},\
	force_think=${force_think},\
	keep_think_below_budget_times=${keep_think_below_budget_times},\
	timeout=${timeout},\
	seed=${seed}\
	" \
	--only_inference

# ignore error with - at the beginning
stop_server:
	@echo "Stopping server on port from config..."
	-bash -c 'PORT=$$(grep "^port:" src/eval/configs/base.yaml | awk "{print \$$2}"); pgrep -f "sglang.launch_server.*--port $$PORT" | xargs -I {} bash -c "pstree -p {} | grep -oP \"\\(\\K[0-9]+(?=\\))\"" | sort -n | uniq | xargs -r kill'
	sleep 10
	@echo "return code $?"