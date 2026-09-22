import os
import signal

import dotenv

# Determine the directory of the current file
current_dir = os.path.dirname(__file__)

# Construct the path to the root directory assuming the structure is known
# If this file is - root/project/module/your_module.py, you want to go up two levels
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

# Construct the .env file path
dotenv_path = os.path.join(project_root, '.env')
print(dotenv_path)

# Load environment variables
dotenv.load_dotenv(dotenv_path=dotenv_path)


from gen_ai_hub.proxy import get_proxy_client
from gen_ai_hub.proxy.native.amazon import Session as BedrockSession
from gen_ai_hub.proxy.native.google_genai import Client as GenAIClient
from gen_ai_hub.proxy.native.openai import chat
from gen_ai_hub.proxy.langchain import OpenAI as ProxyLangchainLLM
from langchain_core.prompts import ChatPromptTemplate

GPT_MODEL_NAME = "gpt-5"
#GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_MODEL_NAME = "gemini-2.5-pro"
ANTHROPIC_MODEL_NAME = "anthropic--claude-4.5-opus"
#ANTHROPIC_MODEL_NAME = "anthropic--claude-4.5-haiku"
MISTRAL_MODEL_NAME = "mistralai--mistral-large-instruct"

MAX_OUTPUT_TOKENS = 8000
CALL_TIMEOUT_SECONDS = 120
MAX_RETRIES = 1

JSON_INSTRUCTION = "Return everything as json."


def timeout_handler(signum, frame):
    raise TimeoutError()


def _call_with_timeout(func, args, label):
    """Call func(*args), aborting via SIGALRM after CALL_TIMEOUT_SECONDS, retrying up to MAX_RETRIES times on timeout."""
    retry_count = 0

    while retry_count < MAX_RETRIES:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(CALL_TIMEOUT_SECONDS)
        try:
            return func(*args)
        except TimeoutError:
            print(f"{label} call timed out. Retrying...")
            retry_count += 1
        except Exception as ex:
            print(f"An unexpected error occurred in {label}: {ex}. Proceeding to next iteration.")
            return None
        finally:
            signal.alarm(0)  # disable alarm whether we succeeded, timed out, or errored

    print("Max retries exceeded. Proceeding to next iteration.")
    return None


def generate_gemini(system_prompt, examples, user_prompt, temperature, response_format=True):
    proxy_client = get_proxy_client("gen-ai-hub")
    client = GenAIClient(proxy_client=proxy_client)

    system_instruction = f"""
    {system_prompt}
    Here's an example:
    {examples}
        """

    config = {
        "system_instruction": system_instruction,
        "temperature": temperature,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "thinking_config": {
            "thinking_budget": 1024,
        },
    }

    if response_format:
        config["response_mime_type"] = "application/json"

    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=user_prompt,
        config=config,
    )
    return response.candidates[0].content.parts[0].text


def generate_gpt(system_prompt, user_prompt, assistant_prompt, prompt, temperature, response_format=True):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_prompt},
        {"role": "user", "content": prompt},
    ]

    kwargs = dict(
        model_name=GPT_MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=MAX_OUTPUT_TOKENS,
        reasoning_effort="low",
    )
    if response_format:
        kwargs["response_format"] = {"type": "json_object"}

    response = chat.completions.create(**kwargs)
    return response.choices[0].message.content


def generate_anthropic(system_prompt, user_prompt, assistant_prompt, prompt, temperature, response_format=True):
    bedrock = BedrockSession().client(model_name=ANTHROPIC_MODEL_NAME)

    full_system_prompt = system_prompt
    if response_format:
        full_system_prompt = f"{system_prompt}\n{JSON_INSTRUCTION}"

    conversation = [
        {"role": "user", "content": [{"text": user_prompt}]},
        {"role": "assistant", "content": [{"text": assistant_prompt}]},
        {"role": "user", "content": [{"text": prompt}]},
    ]

    response = bedrock.converse(
        system=[{"text": full_system_prompt}],
        messages=conversation,
        inferenceConfig={"maxTokens": MAX_OUTPUT_TOKENS, "temperature": temperature},
    )
    return response["output"]["message"]["content"][0]["text"]


def generate_mistral( system_prompt, user_prompt, assistant_prompt, prompt, temperature, response_format=True,):
    proxy_client = get_proxy_client("gen-ai-hub")

    llm = ProxyLangchainLLM(
        proxy_model_name=MISTRAL_MODEL_NAME,
        proxy_client=proxy_client,
        temperature=temperature,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    full_system_prompt = f"{system_prompt} Example: {user_prompt}. Answer: {assistant_prompt}"

    if response_format:
        full_system_prompt += f"\n{JSON_INSTRUCTION}"

    full_system_prompt = full_system_prompt.replace("{", "{{").replace("}", "}}")
    myprompt = ChatPromptTemplate.from_messages([
        ( "system", full_system_prompt ),
        ( "human", "Question: {user_question}"),
    ])

    print(full_system_prompt)
    print(user_prompt)

    llm_chain = myprompt | llm

    response = llm_chain.invoke({
        "user_question": prompt
    })

    return response.strip()

def generate_gemini_with_timeout(system_prompt, examples, user_prompt, temperature, response_format=True):
    return _call_with_timeout(
        generate_gemini, (system_prompt, examples, user_prompt, temperature, response_format), "generate_gemini"
    )

def generate_gpt_with_timeout(system_prompt, user_prompt, assistant_prompt, prompt, temperature, response_format=True):
    return _call_with_timeout(
        generate_gpt, (system_prompt, user_prompt, assistant_prompt, prompt, temperature, response_format), "generate_gpt"
    )

def generate_anthropic_with_timeout(system_prompt, user_prompt, assistant_prompt, prompt, temperature, response_format=True):
    return _call_with_timeout(
        generate_anthropic, (system_prompt, user_prompt, assistant_prompt, prompt, temperature, response_format), "generate_anthropic"
    )

def generate_mistral_with_timeout(system_prompt, user_prompt, assistant_prompt, prompt, temperature, response_format=True):
    return _call_with_timeout(
        generate_mistral, (system_prompt, user_prompt, assistant_prompt, prompt, temperature, response_format), "generate_mistral"
    )
