import os
import dotenv

dotenv.load_dotenv()
from pdb import set_trace as breakpoint
import json
from datetime import datetime, timedelta
from pathlib import Path

import click
import numpy as np
import openai
import pandas as pd
import tqdm
from extract_format import extract_answer
from jinja2 import Template
from omegaconf import OmegaConf
from score import huatuo_match_choice, score
from sglang.utils import (
    launch_server_cmd,
    terminate_process,
    wait_for_server,
)
from transformers import AutoTokenizer


@click.command()
@click.option(
    "--config_path",
    "-c",
    type=str,
    default="src/eval/configs/base.yaml",
)
@click.option(
    "--update_config_by_dotlist",
    "-u",
    type=str,
    default=None,
    help="Update config by dotlist",
)
@click.option("--debug", is_flag=True)
@click.option("--dry_run", is_flag=True)
@click.option("--only_start_server", is_flag=True)
@click.option("--only_inference", is_flag=True)
def main(
    config_path=None,
    update_config_by_dotlist=None,
    debug=False,
    dry_run=False,
    only_start_server=False,
    only_inference=False,
):

    config = OmegaConf.load(config_path)
    if update_config_by_dotlist is not None:
        update_config_by_dotlist = update_config_by_dotlist.split(",")
        update_config_by_dotlist = OmegaConf.from_dotlist(update_config_by_dotlist)
        config = OmegaConf.merge(config, update_config_by_dotlist)
    print(OmegaConf.to_yaml(config))

    output_dir = Path(config.output_dir)
    if only_start_server:
        print("Only start server")
        output_dir = output_dir / "start_server"
    output_dir = output_dir / config.exp_name
    if not only_start_server and output_dir.exists():
        print("exp already exists.")
        if config.overwrite:
            print("Overwrite exp with new version.")
        else:
            print("Not overwrite exp, exit.")
            return

    output_dir, version = prepare_version_dir(output_dir, mkdir=True)
    config.version = version
    print(f"output_dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    start_time = datetime.now()
    log_path = output_dir / "log.txt"
    with open(log_path, "w") as f:
        f.write(f"Start time: {start_time}\n")

    if only_inference is True and only_start_server is True:
        raise ValueError(
            "only_inference and only_start_server cannot be True at the same time"
        )

    sglang_server = SGLangServer(config)
    if dry_run:
        print("Not start server in dry run")
    elif only_inference:
        print("Not start server in only_inference mode")
    else:
        sglang_server.start()

    if only_start_server:
        # NOTE: Stop the server with `bash src/eval/kill_sglang_server.sh`
        print(
            "Only start server, exit. Stop the server with `bash src/eval/kill_sglang_server.sh`"
        )
        return

    client = openai.Client(
        base_url=f"http://127.0.0.1:{config.port}/v1", api_key="EMPTY"
    )
    if config.use_chat_template:
        tokenizer_path = config.tokenizer_path
        if tokenizer_path is None:
            tokenizer_path = config.model_path
        print(f"Use tokenizer_path: {tokenizer_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, padding_side="left", fix_mistral_regex=True
        )

        template = Template(tokenizer.chat_template)
    else:
        raise NotImplementedError("Not implemented")

    if debug:
        responses = call_model(
            ["A mother brings her 3-week-old infant to the pediatrician's office because she is concerned about his feeding habits. He was born without complications and has not had any medical problems up until this time. However, for the past 4 days, he has been fussy, is regurgitating all of his feeds, and his vomit is yellow in color. On physical exam, the child's abdomen is minimally distended but no other abnormalities are appreciated. Which of the following embryologic errors could account for this presentation?\nA. Abnormal hypertrophy of the pylorus\nB. Failure of lateral body folds to move ventrally and fuse in the midline\nC. Abnormal migration of ventral pancreatic bud\nD. Complete failure of proximal duodenum to recanalize"],
            client,
            config,
            template=template,
            tokenizer=tokenizer,
        )
        # fmt: off
        import IPython; IPython.embed()
        # fmt: on
        return

    # Load data
    check_md5(config.eval_data_path, config.eval_data_md5sum)
    input_data = load_huatuo_eval_data(config.eval_data_path)

    if config.limit > 0:
        print(f"limit: {config.limit}")
        input_data = input_data[: config.limit]

    # Run inference
    # NOTE: Format Instruction "Return your final response within \\boxed{}.", but it is not used here.
    query_prompt = "{question}\n{option_str}"
    # breakpoint()
    if config.prefix_prompt is not None:
        prefix_prompt_delimiter = config.prefix_prompt_delimiter
        print(
            f"Add prefix_prompt: {config.prefix_prompt}, delimiter: {prefix_prompt_delimiter}"
        )
        query_prompt = config.prefix_prompt + prefix_prompt_delimiter + query_prompt
    if config.suffix_prompt is not None:
        suffix_prompt_delimiter = config.suffix_prompt_delimiter
        print(
            f"Add suffix_prompt: {config.suffix_prompt}, delimiter: {suffix_prompt_delimiter}"
        )
        query_prompt = query_prompt + suffix_prompt_delimiter + config.suffix_prompt

    print(f"query_prompt: {query_prompt}")

    result_dict_list = []
    for start_idx in tqdm.tqdm(
        range(0, len(input_data), config.batch_size), desc="Inference batches"
    ):
        print("Start inference")
        batch_data = input_data[start_idx : start_idx + config.batch_size]
        print("Fetched batch")
        for item in batch_data:
            item["option_str"] = "\n".join(
                [f"{op}. {ans}" for op, ans in item["options"].items()]
            )
            item["input_str"] = query_prompt.format_map(item)

        prompts = [item["input_str"] for item in batch_data]
        print("Data prepared")
        responses = call_model(
            prompts,
            client,
            config,
            template=template,
            tokenizer=tokenizer,
        )
        print("Model Called")
        for sample, response in zip(batch_data, responses):
            # check
            thinking_text = response["thinking_text"]
            thinking_finish_reason = response["thinking_finish_reason"]

            response_text = response["response_text"]
            finish_reason = response["finish_reason"]

            extracted_answer = extract_answer(response_text)
            huatuo_extracted_answer = huatuo_match_choice(
                response_text, sample["option_str"]
            )
            num_gen_tokens = response["num_gen_tokens"]

            # breakpoint()

            sample["thinking_text"] = thinking_text
            sample["thinking_finish_reason"] = thinking_finish_reason

            sample["response_text"] = response_text
            sample["finish_reason"] = finish_reason

            sample["extracted_answer"] = extracted_answer
            sample["huatuo_extracted_answer"] = huatuo_extracted_answer
            sample["num_gen_tokens"] = num_gen_tokens
            sample["num_keep_think_below_budget"] = response[
                "num_keep_think_below_budget"
            ]
            result_dict_list.append(sample)

    output_path = output_dir / Path(config.eval_data_path).with_suffix(".json").name

    with open(output_path, "w") as f:
        json.dump(result_dict_list, f, indent=2)
    print(f"Save results to {output_path}")

    metrics, mapped_results = score(result_dict_list)

    output_result_json_path = output_path.with_suffix(".scored.json")
    mapped_results.to_json(output_result_json_path, indent=2)
    print(f"Scored results saved to {output_result_json_path}")

    metrics_output_path = output_dir / "metrics.csv"
    pd.DataFrame(metrics).to_csv(metrics_output_path, sep="\t")
    print(f"Metrics saved to {metrics_output_path}")

    # NOTE: terminate sglang server
    if not only_inference:
        sglang_server.terminate()

    with open(log_path, "a") as f:
        end_time = datetime.now()

        f.write(f"End time: {end_time}\n")
        elapsed_time = end_time - start_time
        hours, reminder = divmod(elapsed_time.seconds, 3600)
        minutes, seconds = divmod(reminder, 60)
        f.write(
            f"Script runtime: {int(hours):02}:{int(minutes):02}:{int(seconds):02}\n"
        )


def load_huatuo_eval_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    input_data = []
    if isinstance(data, list):
        data = {"normal": data}
    for k, v in data.items():
        if k=="HumanityLastExam":
            # skip HumanityLastExam, as the original paper does.
            continue
        for da in v:
            da["source"] = k
        input_data.extend(v)
    return input_data


def check_md5(file_path, validation_md5):
    import hashlib

    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    md5sum = hash_md5.hexdigest()
    if md5sum != validation_md5:
        raise ValueError(f"MD5 mismatch: {md5sum} != validation_md5 {validation_md5}")


def call_model(
    prompts,
    client: openai.Client,
    config,
    template=None,
    tokenizer=None,
):
    if config.print_example:
        print("Example:")
        print(prompts[0])

    if config.use_chat_template:
        prompts = [
            template.render(
                messages=[{"role": "user", "content": prom}],
                bos_token=tokenizer.bos_token,
                add_generation_prompt=True,
            )
            for prom in prompts
        ]

    if config.max_tokens > 0:
        new_prompts = []
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(input_ids) > config.max_tokens:
                input_ids = input_ids[: config.max_tokens]
                new_prompts.append(tokenizer.decode(input_ids))
            else:
                new_prompts.append(prompt[-config.max_tokens :])
        prompts = new_prompts

    stop = None
    if config.force_think:
        # https://github.com/simplescaling/s1
        prompts = [i + config.think_str for i in prompts]
        stop = ["<|im_end|>", "<|im_start|>"]
    # NOTE api: https://platform.openai.com/docs/api-reference/completions/create
    response = client.completions.create(
        model="default",
        prompt=prompts,
        temperature=config.temperature,
        # top_p=0.9,
        max_tokens=config.max_new_tokens,
        stop=stop,
        frequency_penalty=config.frequency_penalty,
        # n=1
        timeout=config.timeout,
    )
    # breakpoint()
    results = []
    for _response in response.choices:
        results.append(
            {
                "response_text": _response.text,
                "finish_reason": _response.finish_reason,
                "num_gen_tokens": len(
                    tokenizer.encode(_response.text, add_special_tokens=False)
                ),
                "thinking_text": None,
                "thinking_finish_reason": None,
                "num_keep_think_below_budget": 0,
            }
        )

    results = keep_think(prompts, client, config, tokenizer, stop, results)

    # End thinking
    if config.force_think:
        results = end_thinking(prompts, client, config, tokenizer, stop, results)

    return results


def keep_think(prompts, client, config, tokenizer, stop, results):
    keep_think_below_budget_times = config.keep_think_below_budget_times

    if keep_think_below_budget_times < 0:
        raise ValueError(
            f"Invalid keep_think_below_budget_times: {keep_think_below_budget_times}, should >= 0"
        )

    num_keep_think_below_budget = 0
    while True:
        if num_keep_think_below_budget >= keep_think_below_budget_times:
            break

        final_prompts = []
        keep_think_result_idx_list = []
        num_gen_tokens_list = []
        for idx, (prompt, result) in enumerate(zip(prompts, results)):
            num_gen_tokens = result["num_gen_tokens"]
            response_text = result["response_text"]

            if num_gen_tokens >= config.max_new_tokens:
                continue

                # NOTE: add "Wait"
            keep_think_below_budget_str = config.keep_think_below_budget_str
            if not response_text.endswith("\n") and response_text.endswith(" "):
                keep_think_below_budget_str = " " + keep_think_below_budget_str

            response_text += keep_think_below_budget_str
            num_gen_tokens += len(
                tokenizer.encode(keep_think_below_budget_str, add_special_tokens=False)
            )
            # NOTE: Corner case: when add one word, reach the limit
            if num_gen_tokens >= config.max_new_tokens:
                continue

                # NOTE: update response_text
            result["response_text"] = response_text
            keep_think_result_idx_list.append(idx)
            final_prompts.append(prompt + response_text)
            num_gen_tokens_list.append(num_gen_tokens)

        if len(final_prompts) == 0:
            break

        min_num_gen_tokens = min(num_gen_tokens_list)
        response = client.completions.create(
            model="default",
            prompt=final_prompts,
            temperature=config.temperature,
            # top_p=0.9,
            max_tokens=config.max_new_tokens - min_num_gen_tokens,
            stop=stop,
            frequency_penalty=config.frequency_penalty,
            # n=1
        )

        for selected_idx, _response in zip(
            keep_think_result_idx_list, response.choices
        ):
            results[selected_idx]["response_text"] += _response.text
            results[selected_idx]["finish_reason"] = _response.finish_reason
            results[selected_idx]["num_gen_tokens"] += len(
                tokenizer.encode(_response.text, add_special_tokens=False)
            )
            results[selected_idx]["num_keep_think_below_budget"] += 1

        num_keep_think_below_budget += 1

    return results


def end_thinking(prompts, client, config, tokenizer, stop, results):
    final_prompts = []
    # https://github.com/simplescaling/s1
    # https://github.com/simplescaling/s1/blob/main/eval/lm-evaluation-harness/lm_eval/models/vllm_causallms.py
    for prompt, result in zip(prompts, results):
        answer_prefix = ""
        thinking_text = result["response_text"]
        thinking_finish_reason = result["finish_reason"]
        num_gen_tokens = result["num_gen_tokens"]

        if not thinking_text.endswith("\n"):
            thinking_text += "\n"
            num_gen_tokens += len(tokenizer.encode("\n", add_special_tokens=False))

        if thinking_finish_reason == "length":
            answer_prefix = config.start_overthink_answer_str
            num_gen_tokens += len(
                tokenizer.encode(answer_prefix, add_special_tokens=False)
            )
        elif thinking_finish_reason == "stop":
            answer_prefix = config.start_answer_str
            num_gen_tokens += len(
                tokenizer.encode(answer_prefix, add_special_tokens=False)
            )
        else:
            raise ValueError(f"Invalid finish_reason: {thinking_finish_reason}")

        result["response_text"] = None
        result["finish_reason"] = None
        result["thinking_text"] = thinking_text
        result["thinking_finish_reason"] = thinking_finish_reason
        result["num_gen_tokens"] = num_gen_tokens
        result["answer_prefix"] = answer_prefix

        final_prompts.append(prompt + thinking_text + answer_prefix)

    response = client.completions.create(
        model="default",
        prompt=final_prompts,
        temperature=config.temperature,
        # top_p=0.9,
        max_tokens=config.max_new_answer_tokens,
        stop=stop,
        frequency_penalty=config.frequency_penalty,
        # n=1
    )

    for result, _response in zip(results, response.choices):
        answer_prefix = result["answer_prefix"]
        result["response_text"] = answer_prefix + _response.text
        result["finish_reason"] = _response.finish_reason
        result["num_gen_tokens"] += len(
            tokenizer.encode(_response.text, add_special_tokens=False)
        )

    return results


class SGLangServer:
    # NOTE start sglang server
    # https://docs.sglang.ai/backend/send_request.html
    def __init__(self, config):
        self.config = config

        self.model_path = config.model_path
        self.port = config.port
        self.dp = config.dp
        self.tp = config.tp
        self.mem_fraction_static = config.mem_fraction_static
        self.seed = config.seed
        self.log_level = config.log_level

        self.server_process = None

    def start(self):
        # NOTE: sglang api, https://github.com/sgl-project/sglang/blob/10b544ae9b426c0b081cf06e5fcd1f24f82d7443/docs/backend/patch.py#L28
        # NOTE: deterministic is still under construction. See:
        # https://github.com/sgl-project/sglang/issues/4042 "Others"
        # https://github.com/sgl-project/sglang/issues/1335
        # https://docs.sglang.ai/references/faq.html

        server_process, port = launch_server_cmd(
            f"""
        python -m sglang.launch_server \
        --model-path {self.model_path} \
        --mem-fraction-static {self.mem_fraction_static} \
        --dp {self.dp} \
        --tp {self.tp} \
        --random-seed {self.seed} \
        --log-level {self.log_level}
        """,
            port=self.port,
        )
        if port != self.port:
            self.terminate()
            raise ValueError(f"Port mismatch: return {port} != set {self.port}")

        wait_for_server(f"http://localhost:{port}")

        print(f"start sglang server at port {port}")
        self.server_process = server_process

    def terminate(self):
        if self.server_process is None:
            raise ValueError("Server is not running")

        terminate_process(self.server_process)


def _get_next_version(base_dir) -> int:
    versions_root = Path(base_dir)
    versions_root.mkdir(parents=True, exist_ok=True)

    if not versions_root.is_dir():
        print("Missing logger folder: %s", versions_root)
        return 0

    existing_versions = []
    for d in versions_root.iterdir():
        if d.is_dir() and d.name.startswith("version_"):
            dir_ver = d.name.split("_")[1]
            if dir_ver.isdigit():
                existing_versions.append(int(dir_ver))

    if len(existing_versions) == 0:
        return 0

    return max(existing_versions) + 1


def prepare_version_dir(base_dir, mkdir=False):
    base_dir = Path(base_dir)
    version = _get_next_version(base_dir)
    version_dir = base_dir / f"version_{version}"
    if mkdir:
        version_dir.mkdir(parents=True, exist_ok=True)
    return version_dir, version


if __name__ == "__main__":
    main()
