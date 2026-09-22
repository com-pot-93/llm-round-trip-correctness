import os, sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import warnings
warnings.filterwarnings("ignore")

import json
import logging
import argparse
import csv

sys.path.append("./data/")
sys.path.append("./model_evaluation")
sys.path.append("./round_trip")

import bpmn_similarity
from text_evaluation import text_similarity


def _write_json_atomic(path, data):
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)
    os.replace(tmp_path, path)


def evaluation_pipeline(llm, direction, model_path, text_path, example):
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger("{}EvalLogger".format(llm.upper()))
    logger.setLevel(logging.INFO)

    artefacts_path = './generated_artefacts/report_{}_{}_{}.json'.format(llm, example, direction)
    try:
        with open(artefacts_path, 'r') as infile:
            artefacts = json.load(infile)
    except FileNotFoundError:
        logger.error(f"No generated artefacts found at {artefacts_path}. Run generation_pipeline.py first.")
        return

    os.makedirs('./results', exist_ok=True)
    results_path = './results/{}_{}_{}.csv'.format(llm, example, direction)
    with open(results_path, 'w', newline='') as csvfile:
        csv.writer(csvfile).writerow(['model_name', 't2t_eval_1', 't2t_eval_2', 'm2m_eval_1', 'm2m_eval_2'])

    os.makedirs('./iter_results', exist_ok=True)
    iterations_path = './iter_results/report_{}_{}_{}.json'.format(llm, example, direction)
    iteration_results = {}
    _write_json_atomic(iterations_path, iteration_results)

    logger.info('Starting evaluation of generated artefacts')

    for i, (file, iterations) in enumerate(artefacts.items()):
        logger.info(f'Evaluating file: {file}')
        print('--------------------------{}/{}-----------------------------------'.format(i, len(artefacts)))
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

            text_eval_1 = []
            text_eval_2 = []
            model_eval_1 = []
            model_eval_2 = []
            iteration_results[file] = {}

            for j, artefact in iterations.items():
                gen_text = artefact.get('text')
                gen_model = artefact.get('model')
                iteration_scores = {'text_eval_1': 'N/A', 'text_eval_2': 'N/A', 'model_eval_1': 'N/A', 'model_eval_2': 'N/A'}

                if description and gen_text:
                    try:
                        iteration_scores['text_eval_1'] = text_similarity.sts_bert(description, gen_text)
                        iteration_scores['text_eval_2'] = text_similarity.text_similarity_alternative(description, gen_text, threshold=0.7)
                        text_eval_1.append(iteration_scores['text_eval_1'])
                        text_eval_2.append(iteration_scores['text_eval_2'])
                    except Exception as e:
                        logger.error(f"Error during text evaluation in iteration {j} for file {file}: {e}")

                if model and gen_model:
                    try:
                        scores, scores_alt = bpmn_similarity.calculate_similarity(
                            model, gen_model, method="dice", similarity_threshold=0.7
                        )
                        iteration_scores['model_eval_1'] = scores["overall"]
                        iteration_scores['model_eval_2'] = scores_alt["overall"]
                        model_eval_1.append(iteration_scores['model_eval_1'])
                        model_eval_2.append(iteration_scores['model_eval_2'])
                    except Exception as e:
                        logger.error(f"Error during model evaluation in iteration {j} for file {file}: {e}")

                iteration_results[file][j] = iteration_scores

            with open(results_path, 'a', newline='') as csvfile:
                csv.writer(csvfile).writerow([
                    file,
                    sum(text_eval_1) / len(text_eval_1) if text_eval_1 else 'N/A',
                    sum(text_eval_2) / len(text_eval_2) if text_eval_2 else 'N/A',
                    sum(model_eval_1) / len(model_eval_1) if model_eval_1 else 'N/A',
                    sum(model_eval_2) / len(model_eval_2) if model_eval_2 else 'N/A',
                ])
            _write_json_atomic(iterations_path, iteration_results)

        except Exception as e:
            logger.error(f"An error occurred while evaluating file {file}: {e}")

    logger.info('Completed evaluation of all generated artefacts')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate previously generated artefacts against ground truth, without generating anything.")
    parser.add_argument('--llm', type=str, required=True, help='Which llm generated the artefacts: gpt, gemini, anthropic or mistral')
    parser.add_argument('--model-path', type=str, required=False, default='no', help='Path to the ground-truth models directory')
    parser.add_argument('--text-path', type=str, required=False, default='no', help='Path to the ground-truth text descriptions directory')
    parser.add_argument('--example', type=str, required=True, help='Dataset name used when the artefacts were generated')
    parser.add_argument('--direction', type=str, required=True, help='m2m or t2t (must match the generation run)')

    args = parser.parse_args()
    llm = args.llm.lower()
    direction = args.direction.lower()
    model_path = args.model_path if os.path.isdir(args.model_path) else 'no'
    text_path = args.text_path if os.path.isdir(args.text_path) else 'no'
    example = args.example.lower()

    if direction not in ('m2m', 't2t'):
        print('Please check the direction of the pipeline (only m2m or t2t are acceptable)!!!')
    else:
        evaluation_pipeline(llm, direction, model_path, text_path, example)

