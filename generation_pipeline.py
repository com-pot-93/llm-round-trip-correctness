import os, sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import warnings
warnings.filterwarnings("ignore")

import json
import logging
import argparse
import time
from functools import partial

sys.path.append("./data/")
sys.path.append("./model_evaluation")
sys.path.append("./round_trip")

from round_trip.t2m.prompt_engineering import json_desc
from round_trip.m2t.create_description import generate_prompt_gpt as generate_prompt_gpt_m2t
from round_trip.t2m.create_model import generate_prompt_gpt as generate_prompt_gpt_t2m
from round_trip.m2t.create_description import generate_prompt_gemini as generate_prompt_gemini_m2t
from round_trip.t2m.create_model import generate_prompt_gemini as generate_prompt_gemini_t2m

from round_trip.llm_connect.gen_ai_llm_call import (
    generate_gpt_with_timeout,
    generate_gemini_with_timeout,
    generate_anthropic_with_timeout,
    generate_mistral_with_timeout,
)


EXAMPLE_PATHS = {
    'pet': ("./data/prompt_ex_json_pet.json", "./data/prompt_ex_text_pet.txt"),
    'real_set': ("./data/prompt_ex_json_real_set.json", "./data/prompt_ex_text_real_set.txt"),
}
DEFAULT_EXAMPLE_PATH = ("./data/prompt_ex_json_sapsam.json", "./data/prompt_ex_text_sapsam.txt")

# gpt, anthropic and mistral all take the same (system, user, assistant, prompt, temp, response_format)
# call shape, so they share one prompt-generation path; only gemini's shape differs.
GPT_STYLE_CALLS = {
    'gpt': generate_gpt_with_timeout,
    'anthropic': generate_anthropic_with_timeout,
    'mistral': generate_mistral_with_timeout,
}
SUPPORTED_LLMS = {'gemini', *GPT_STYLE_CALLS}


class Prompt:
    """Wraps an LLM's t2m/m2t prompts behind a uniform call_t2m/call_m2t(text, temperature, response_format) interface."""

    def __init__(self, llm, path_to_json, path_to_text, json_desc, temp_in, temp_out):
        self.temp_in = temp_in
        self.temp_out = temp_out

        if llm == 'gemini':
            system_prompt_t2m, examples_t2m = generate_prompt_gemini_t2m(path_to_json, path_to_text, json_desc)
            system_prompt_m2t, examples_m2t = generate_prompt_gemini_m2t(path_to_json, path_to_text)
            self.call_t2m = partial(generate_gemini_with_timeout, system_prompt_t2m, examples_t2m)
            self.call_m2t = partial(generate_gemini_with_timeout, system_prompt_m2t, examples_m2t)
        elif llm in GPT_STYLE_CALLS:
            system_prompt_t2m, user_prompt_t2m, assistant_prompt_t2m = generate_prompt_gpt_t2m(path_to_json, path_to_text, json_desc)
            system_prompt_m2t, user_prompt_m2t, assistant_prompt_m2t = generate_prompt_gpt_m2t(path_to_json, path_to_text)
            call = GPT_STYLE_CALLS[llm]
            self.call_t2m = partial(call, system_prompt_t2m, user_prompt_t2m, assistant_prompt_t2m)
            self.call_m2t = partial(call, system_prompt_m2t, user_prompt_m2t, assistant_prompt_m2t)
        else:
            raise ValueError(f"Unsupported llm: {llm}")


def _write_json_atomic(path, data):
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)
    os.replace(tmp_path, path)


def generate_artefacts(prompt, direction, model, description):
    """Round-trips model<->text through the LLM. 'm2m' goes model->text->model, 't2t' goes text->model->text."""
    gen_text = ''
    gen_model = ''

    if direction == 'm2m':
        gen_text = prompt.call_m2t("Here is the model: " + str(model), prompt.temp_in, response_format=False)
        if gen_text:
            gen_model = prompt.call_t2m("Here is the textual description: " + gen_text, prompt.temp_out, response_format=True)
    elif direction == 't2t':
        gen_model = prompt.call_t2m("Here is the textual description: " + str(description), prompt.temp_in, response_format=True)
        if gen_model:
            gen_text = prompt.call_m2t("Here is the model: " + str(gen_model), prompt.temp_out, response_format=False)

    return gen_text, gen_model


def generation_pipeline(llm, direction, model_path, text_path, example):
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger("{}Logger".format(llm.upper()))
    logger.setLevel(logging.INFO)
    path_to_json, path_to_text = EXAMPLE_PATHS.get(example, DEFAULT_EXAMPLE_PATH)

    temp_in = 1
    temp_out = 0
    iterations = 10

    prompt = Prompt(llm, path_to_json, path_to_text, json_desc, temp_in, temp_out)

    if direction == 'm2m':
        files_to_iterate = os.listdir(model_path)
    elif direction == 't2t':
        files_to_iterate = os.listdir(text_path)

    os.makedirs('./generated_artefacts', exist_ok=True)
    artefacts_path = './generated_artefacts/report_{}_{}_{}.json'.format(llm, example, direction)
    artefacts = {}
    _write_json_atomic(artefacts_path, artefacts)

    logger.info('Starting the processing of models and texts')

    for i, file in enumerate(files_to_iterate):  # Use enumerate for progress tracking
        logger.info(f'Processing file: {file}')
        print('--------------------------{}/{}-----------------------------------'.format(i, len(files_to_iterate)))
        try:
            if model_path != 'no':
                split_file = '{}.json'.format(file.split('.')[0])
                with open(os.path.join(model_path, split_file), "r") as infile:
                    model = json.load(infile)
            else:
                model = ''

            if text_path != 'no':
                split_file = '{}.txt'.format(file.split('.')[0])
                with open(os.path.join(text_path, split_file), "r") as infile:
                    description = infile.read()
            else:
                description = ''

            artefacts[file] = {}

            for j in range(iterations):
                logger.info(f'Iteration {j + 1} for file: {file}')
                start_time = time.perf_counter()
                gen_text, gen_model = generate_artefacts(prompt, direction, model, description)
                running = time.perf_counter() - start_time
                logger.info(f'TIME {running} for file: {file}')

                if not gen_text or not gen_model:
                    continue

                try:
                    gen_model_json = json.loads(gen_model)
                except Exception as e:
                    logger.error(f"Error parsing generated model in iteration {j + 1} for file {file}: {e}")
                    continue

                artefacts[file][j] = {'text': gen_text, 'model': gen_model_json}

            _write_json_atomic(artefacts_path, artefacts)

        except Exception as e:
            logger.error(f"An error occurred while processing file {file}: {e}")

    logger.info('Completed processing all models and texts')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate round-tripped models/text via an llm, with retries and timeout.")
    parser.add_argument('--llm', type=str, required=True, help='Select llm mode: {}'.format(', '.join(sorted(SUPPORTED_LLMS))))
    parser.add_argument('--model-path', type=str, required=False, default='no', help='Path to the models directory')
    parser.add_argument('--text-path', type=str, required=False, default='no', help='Path to the text descriptions directory')
    parser.add_argument('--example', type=str, required=True, help='pet or real_set')
    parser.add_argument('--direction', type=str, required=True, help='m2m or t2t')

    args = parser.parse_args()
    llm = args.llm.lower()
    direction = args.direction.lower()
    model_path = args.model_path
    text_path = args.text_path
    example = args.example.lower()

    if llm not in SUPPORTED_LLMS:
        print('Please check selected llm model (only {} are acceptable)!!!'.format(', '.join(sorted(SUPPORTED_LLMS))))
    elif direction == 'm2m' and os.path.isdir(model_path):
        if not os.path.isdir(text_path):
            text_path = 'no'
        generation_pipeline(llm, direction, model_path, text_path, example)
    elif direction == 't2t' and os.path.isdir(text_path):
        if not os.path.isdir(model_path):
            model_path = 'no'
        generation_pipeline(llm, direction, model_path, text_path, example)
    else:
        print('Please check the direction of the pipeline (only m2m or t2t are acceptable) or check provided directories!!!')

