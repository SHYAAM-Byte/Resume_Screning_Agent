
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import ollama

from Prompts import (
    SYSTEM_PROMPT,
    build_resume_extraction_prompt,
    build_comparison_prompt,
)
from Utils import safe_json_parse

logger = logging.getLogger(__name__)

JSONResult = Union[Dict[str, Any], List[Any]]

DEFAULT_MODEL = "llama3"
DEFAULT_MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 0.2  # Low temperature favors consistent, structured output


class AIEngineError(Exception):
    """Base exception for all ai_engine failures."""


class OllamaConnectionError(AIEngineError):
    """Raised when the Ollama server/model cannot be reached or used."""


class InvalidLLMResponseError(AIEngineError):
    """Raised when the LLM fails to return valid JSON after all retries."""


def _call_ollama(
    prompt: str,
    system_prompt: Optional[str],
    model: str,
    temperature: float,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            format="json",
            options={"temperature": temperature},
        )
    except ollama.ResponseError as exc:
        raise OllamaConnectionError(
            f"Ollama could not fulfill the request for model '{model}'. "
            f"If the model isn't pulled yet, run: ollama pull {model}. "
            f"Original error: {exc}"
        ) from exc
    except ollama.RequestError as exc:
        raise OllamaConnectionError(f"Invalid request sent to Ollama: {exc}") from exc
    except Exception as exc:
        raise OllamaConnectionError(
            f"Could not reach Ollama. Make sure the server is running "
            f"('ollama serve') and reachable. Original error: {exc}"
        ) from exc

    content = getattr(getattr(response, "message", None), "content", None)
    if content is None:
        raise OllamaConnectionError(f"Unexpected response shape from Ollama: {response!r}")

    return content

def generate_json_response(
    prompt: str,
    system_prompt: Optional[str] = SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    temperature: float = DEFAULT_TEMPERATURE,
) -> JSONResult:
    last_raw_response = ""
    current_prompt = prompt

    for attempt in range(1, max_retries + 1):
        last_raw_response = _call_ollama(
            prompt=current_prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
        )

        parsed = safe_json_parse(last_raw_response)
        if parsed is not None:
            return parsed

        logger.warning(
            "Attempt %d/%d: model '%s' returned invalid JSON. Retrying...",
            attempt, max_retries, model,
        )

        # Escalate the reminder on retry, but keep the original prompt intact.
        current_prompt = (
            f"{prompt}\n\n"
            f"REMINDER: Your previous response was not valid JSON. "
            f"Respond with ONLY a valid JSON object and nothing else — "
            f"no markdown, no code fences, no commentary."
        )

    raise InvalidLLMResponseError(
        f"Model '{model}' failed to return valid JSON after {max_retries} attempts. "
        f"Last raw response: {last_raw_response[:500]!r}"
    )


def extract_resume_info(
    resume_text: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> JSONResult:
    prompt = build_resume_extraction_prompt(resume_text)
    return generate_json_response(prompt, model=model, max_retries=max_retries)


def compare_resume_to_job(
    resume_data: str,
    job_description: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> JSONResult:
    prompt = build_comparison_prompt(resume_data, job_description)
    return generate_json_response(prompt, model=model, max_retries=max_retries)

# from __future__ import annotations

# import json
# import logging
# import os
# from typing import Any, Dict, List, Optional, Union

# from dotenv import load_dotenv
# from openai import OpenAI

# from Prompts import (
#     SYSTEM_PROMPT,
#     build_resume_extraction_prompt,
#     build_comparison_prompt,
# )
# from Utils import safe_json_parse

# load_dotenv()

# logger = logging.getLogger(__name__)

# JSONResult = Union[Dict[str, Any], List[Any]]

# # --------------------------------------------------------------------
# # Configuration
# # --------------------------------------------------------------------

# DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
# DEFAULT_MAX_RETRIES = 3
# DEFAULT_TEMPERATURE = 0.2

# client = OpenAI(
#     api_key=os.getenv("OLLAMA_API_KEY"),
#     base_url=os.getenv("OLLAMA_BASE_URL"),
# )

# # --------------------------------------------------------------------
# # Exceptions
# # --------------------------------------------------------------------

# class AIEngineError(Exception):
#     pass


# class OllamaConnectionError(AIEngineError):
#     pass


# class InvalidLLMResponseError(AIEngineError):
#     pass


# # --------------------------------------------------------------------
# # Internal Call
# # --------------------------------------------------------------------

# def _call_model(
#     prompt: str,
#     system_prompt: Optional[str],
#     model: str,
#     temperature: float,
# ) -> str:

#     messages = []

#     if system_prompt:
#         messages.append(
#             {
#                 "role": "system",
#                 "content": system_prompt,
#             }
#         )

#     messages.append(
#         {
#             "role": "user",
#             "content": prompt,
#         }
#     )

#     try:

#         response = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             temperature=temperature,
#             response_format={
#                 "type": "json_object"
#             },
#         )

#         return response.choices[0].message.content

#     except Exception as exc:
#         raise OllamaConnectionError(str(exc))


# # --------------------------------------------------------------------
# # Generic JSON Engine
# # --------------------------------------------------------------------

# def generate_json_response(
#     prompt: str,
#     system_prompt: Optional[str] = SYSTEM_PROMPT,
#     model: str = DEFAULT_MODEL,
#     max_retries: int = DEFAULT_MAX_RETRIES,
#     temperature: float = DEFAULT_TEMPERATURE,
# ) -> JSONResult:

#     current_prompt = prompt
#     last_response = ""

#     for attempt in range(max_retries):

#         last_response = _call_model(
#             current_prompt,
#             system_prompt,
#             model,
#             temperature,
#         )

#         parsed = safe_json_parse(last_response)

#         if parsed is not None:
#             return parsed

#         logger.warning(
#             "Invalid JSON attempt %d/%d",
#             attempt + 1,
#             max_retries,
#         )

#         current_prompt = (
#             prompt
#             + "\n\n"
#             + "IMPORTANT:\n"
#             + "Return ONLY valid JSON."
#         )

#     raise InvalidLLMResponseError(last_response)


# # --------------------------------------------------------------------
# # Resume Extraction
# # --------------------------------------------------------------------

# def extract_resume_info(
#     resume_text: str,
#     model: str = DEFAULT_MODEL,
#     max_retries: int = DEFAULT_MAX_RETRIES,
# ):

#     prompt = build_resume_extraction_prompt(resume_text)

#     return generate_json_response(
#         prompt,
#         model=model,
#         max_retries=max_retries,
#     )


# # --------------------------------------------------------------------
# # Resume Comparison
# # --------------------------------------------------------------------

# def compare_resume_to_job(
#     resume_data: str,
#     job_description: str,
#     model: str = DEFAULT_MODEL,
#     max_retries: int = DEFAULT_MAX_RETRIES,
# ):

#     prompt = build_comparison_prompt(
#         resume_data,
#         job_description,
#     )

#     return generate_json_response(
#         prompt,
#         model=model,
#         max_retries=max_retries,
#     )