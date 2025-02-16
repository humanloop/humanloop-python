from dataclasses import dataclass
import json
from typing import Callable
import pytest

import os
import requests
from humanloop.client import Humanloop
from tests.integration.conftest import APIKeys

from tests.assets import levenshtein, exact_match


@pytest.fixture(scope="session", autouse=True)
def medqa_knowledge_base_path() -> str:
    file_path = "tests/assets/medqa.parquet"
    if not os.path.exists(file_path):
        os.makedirs("tests/assets", exist_ok=True)
        url = "https://github.com/humanloop/humanloop-cookbook/raw/refs/heads/main/assets/sources/textbooks.parquet"
        response = requests.get(url)
        with open(file_path, "wb+") as file:
            file.write(response.content)
    return file_path


@pytest.fixture(scope="session", autouse=True)
def medqa_dataset_path() -> str:
    file_path = "tests/assets/datapoints.jsonl"
    if not os.path.exists(file_path):
        os.makedirs("tests/assets", exist_ok=True)
        url = "https://raw.githubusercontent.com/humanloop/humanloop-cookbook/refs/heads/main/assets/datapoints.jsonl"
        response = requests.get(url)
        with open(file_path, "wb+") as file:
            file.write(response.content)
    return file_path


@dataclass
class MedQAScenario:
    retrieval_tool: tuple[str, Callable[[str], str]]
    call_model: tuple[str, Callable[..., str]]
    ask_question: tuple[str, Callable[..., str]]
    medqa_dataset_path: tuple[str, list[dict]]
    levenshtein_path: str
    exact_match_path: str


@pytest.fixture()
def evaluate_medqa_scenario(
    humanloop_client: "Humanloop",
    get_test_path: Callable[[str], str],
    api_keys: APIKeys,
    medqa_knowledge_base_path: str,
    medqa_dataset_path: str,
) -> MedQAScenario:
    import inspect

    from chromadb import chromadb  # type: ignore
    from openai import OpenAI
    import pandas as pd  # type: ignore

    chroma = chromadb.Client()
    collection = chroma.get_or_create_collection(name="MedQA")
    knowledge_base = pd.read_parquet(medqa_knowledge_base_path)
    knowledge_base = knowledge_base.sample(10, random_state=42)
    collection.add(
        documents=knowledge_base["contents"].to_list(),
        ids=knowledge_base["id"].to_list(),
    )

    openai = OpenAI(api_key=api_keys.openai)

    MODEL = "gpt-4o-mini"
    TEMPLATE = [
        {
            "role": "system",
            "content": """Answer the following question factually.

    Question: {{question}}

    Options:
    - {{option_A}}
    - {{option_B}}
    - {{option_C}}
    - {{option_D}}
    - {{option_E}}

    ---

    Here is some retrieved information that might be helpful.
    Retrieved data:
    {{retrieved_data}}

    ---

    Give you answer in 3 sections using the following format. Do not include the quotes or the brackets. Do include the "---" separators.
    ```
    <chosen option verbatim>
    ---
    <clear explanation of why the option is correct and why the other options are incorrect. keep it ELI5.>
    ---
    <quote relevant information snippets from the retrieved data verbatim. every line here should be directly copied from the retrieved data>
    ```
    """,
        }
    ]

    @humanloop_client.tool(path=get_test_path("Retrieval"))
    def retrieval_tool(question: str) -> str:
        """Retrieve most relevant document from the vector db (Chroma) for the question."""
        response = collection.query(query_texts=[question], n_results=1)
        retrieved_doc = response["documents"][0][0]
        return retrieved_doc

    @humanloop_client.prompt(path=get_test_path("Call Model"))
    def call_model(**inputs):
        """Populate the Prompt template."""
        messages = humanloop_client.prompts.populate_template(TEMPLATE, inputs)

        # Call OpenAI to get response
        chat_completion = openai.chat.completions.create(
            model=MODEL,
            temperature=0,
            presence_penalty=0,
            frequency_penalty=0,
            messages=messages,
        )
        return chat_completion.choices[0].message.content

    @humanloop_client.flow(
        path=get_test_path("Pipeline"),
        attributes={
            "prompt": {
                "template": [
                    {
                        "role": "system",
                        "content": 'Answer the following question factually.\n\nQuestion: {{question}}\n\nOptions:\n- {{option_A}}\n- {{option_B}}\n- {{option_C}}\n- {{option_D}}\n- {{option_E}}\n\n---\n\nHere is some retrieved information that might be helpful.\nRetrieved data:\n{{retrieved_data}}\n\n---\n\nGive you answer in 3 sections using the following format. Do not include the quotes or the brackets. Do include the "---" separators.\n```\n<chosen option verbatim>\n---\n<clear explanation of why the option is correct and why the other options are incorrect. keep it ELI5.>\n---\n<quote relevant information snippets from the retrieved data verbatim. every line here should be directly copied from the retrieved data>\n```\n',
                    }
                ],
                "model_name": "gpt-4o",
                "temperature": 0,
            },
            "tool": {
                "name": "retrieval_tool_v3",
                "description": "Retrieval tool for MedQA.",
                "source_code": inspect.getsource(retrieval_tool),
            },
        },
    )
    def ask_question(**inputs) -> str:
        """Ask a question and get an answer using a simple RAG pipeline"""

        # Retrieve context
        retrieved_data = retrieval_tool(inputs["question"])
        inputs = {**inputs, "retrieved_data": retrieved_data}

        # Call LLM
        return call_model(**inputs)

    with open(medqa_dataset_path, "r") as file:
        datapoints = [json.loads(line) for line in file.readlines()][:20]

    for path, code, return_type in [
        (get_test_path("Levenshtein Distance"), levenshtein, "number"),
        (get_test_path("Exact Match"), exact_match, "boolean"),
    ]:
        humanloop_client.evaluators.upsert(
            path=path,
            # TODO: spec comes up as Any
            spec={
                "arguments_type": "target_required",
                "return_type": return_type,
                "evaluator_type": "python",
                "code": inspect.getsource(code),
            },
        )

    return MedQAScenario(
        retrieval_tool=(get_test_path("Retrieval"), retrieval_tool),
        call_model=(get_test_path("Call Model"), call_model),
        ask_question=(get_test_path("Pipeline"), ask_question),
        medqa_dataset_path=(get_test_path("MedQA Dataset"), datapoints),
        levenshtein_path=get_test_path("Levenshtein Distance"),
        exact_match_path=get_test_path("Exact Match"),
    )
