import dotenv

dotenv.load_dotenv()

import difflib
import json
import re
from pathlib import Path
from pdb import set_trace as breakpoint
import click
import datasets
import openai
import tqdm
from jinja2 import Template
from omegaconf import OmegaConf
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

    output_dir = Path(config.output_dir)
    output_dir = output_dir / Path(config.model_path).stem / config.exp_name
    output_dir, version = prepare_version_dir(output_dir, config.version, mkdir=True)
    config.version = version
    print(f"output_dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    result_json_path = (
        output_dir / Path(config.eval_data_path).with_suffix(".json").name
    )

    print(f"Loading json: {result_json_path}")
    with open(result_json_path) as f:
        results = json.load(f)
    print("Finished loading json")

    metrics, mapped_results = score(results)

    output_result_json_path = result_json_path.with_suffix(".scored.json")
    mapped_results.to_json(output_result_json_path, indent=2)
    print(f"Scored results saved to {output_result_json_path}")

    metrics_output_path = output_dir / "metrics.json"
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_output_path}")


def score(results):
    results = datasets.Dataset.from_list(results)

    def _map_correct(result):
        extracted_answer = result["extracted_answer"]
        answer_letter = result["answer_idx"]
        answer_string = result["answer"]
        source = result["source"]
        question = result["question"]
        option_str = result["option_str"]

        huatuo_extracted_answer = None
        correct = False
        if extracted_answer.lower() == answer_string.lower():
            correct = True
        elif extracted_answer.lower() == f"{answer_letter}. {answer_string}".lower():
            correct = True
        elif (
            len(extracted_answer) == 1
            and extracted_answer.lower() == answer_letter.lower()
        ):
            correct = True
        else:
            huatuo_extracted_answer = huatuo_match_choice(extracted_answer, option_str)
            if huatuo_extracted_answer.lower() == answer_letter.lower():
                correct = True

        return {
            "correct": correct,
            "extracted_answer": extracted_answer,
            "answer": answer_string,
            "source": source,
            "option_str": option_str,
            "question": question,
            "huatuo_extracted_answer": huatuo_extracted_answer,
        }

    mapped_results = results.map(
        _map_correct,
        keep_in_memory=True,
        remove_columns=results.column_names,
    )
    source_list = mapped_results.unique("source")

    metrics = {}
    for source in source_list:
        source_results = mapped_results.filter(lambda x: x["source"] == source)
        correct_count = source_results.filter(lambda x: x["correct"]).num_rows
        total_count = source_results.num_rows
        accuracy = correct_count / total_count
        metrics[source] = {
            "correct_count": correct_count,
            "total_count": total_count,
            "accuracy": accuracy,
        }
    print(f"Metrics:\n{metrics}")

    return metrics, mapped_results


def huatuo_match_choice(text, option_str):
    # From HuatuoGPT-o1 https://github.com/FreedomIntelligence/HuatuoGPT-o1/blob/main/evaluation/eval.py
    options = {}
    for _option in option_str.split("\n"):
        _option = _option.strip()
        # NOTE: only split 1 time
        try:
            option_letter, option_text = _option.split(". ", 1)
        except ValueError as e:
            print(f"Error: {e} with option: {_option}")
            continue
        options[option_letter] = option_text

    # For HuatuoGPT-o1
    if "## Final Response\n\n" in text:
        text = text.split("## Final Response\n\n")[-1]

    # For strict prompt
    matches = list(re.finditer(r"(answer is\s*?)([A-N])", text, re.S))
    if matches:
        # first_match_answer = matches[0].group(2)
        last_match_answer = matches[-1].group(2)
        return last_match_answer

    # Non strict
    match_options = "ABCDEFGHIJKLMN"[: len(options)]
    matches = list(
        re.finditer(
            r"([\u4e00-\u9fff]|is |是|项|\*|\W|\ |\(|为|^|'|\"|#)(?![aA] )(["
            + match_options
            + r"])(\W|[\u4e00-\u9fff]|$)",
            text,
            re.S,
        )
    )
    if matches:
        last_match_answer = matches[-1].group(2)
        return last_match_answer

    # Strictly find option text
    text = text.lower()
    option_letter_text_pairs = [
        (opt, text.rindex(options[opt].lower()))
        for opt in options
        if options[opt].lower() in text
    ]
    if len(option_letter_text_pairs) > 0:
        last_match_answer = sorted(
            option_letter_text_pairs, key=lambda x: x[1], reverse=True
        )[0][0]

        return last_match_answer

    # Fuzzy find option text
    else:
        option_letters = [x for x in options]
        option_texts = [options[x].lower() for x in options]
        most_similar_index = find_most_similar_index(option_texts, text.lower())
        return option_letters[most_similar_index]

        # return text


def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = None
    highest_similarity = 0

    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str, target_str)

        # If the current string is more similar than the previous most similar string, update the variables
        if similarity >= highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity

    return most_similar_index


def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()


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


def prepare_version_dir(base_dir, version=None, mkdir=False):
    base_dir = Path(base_dir)
    if version is None:
        version = _get_next_version(base_dir)
    version_dir = base_dir / f"version_{version}"
    if mkdir:
        version_dir.mkdir(parents=True, exist_ok=True)
    return version_dir, version


if __name__ == "__main__":
    main()
