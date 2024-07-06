import logging
import json
import ast
import os
import time

import numpy as np
from aiohttp import ClientSession
from typing import Dict, List, Optional, Union
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from pydantic import Field

from agentverse.llms.base import LLMResult
from agentverse.logging import logger
from agentverse.message import Message

from . import llm_registry, LOCAL_LLMS, LOCAL_LLMS_MAPPING
from .base import BaseChatModel, BaseModelArgs
from .utils.jsonrepair import JsonRepair
from .utils.llm_server_utils import get_llm_server_modelname

try:
    from groq import Groq, AsyncGroq, GroqError
except ImportError:
    is_groq_available = False
    logger.warn(
        "groq package is not installed. Please install it via `pip install groq`"
    )
else:
    api_key = None
    model_name = None
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

    if not GROQ_API_KEY:
        logger.warn(
            "Groq API key is not set. Please set an environment variable GROQ_API_KEY."
        )
    else:
        DEFAULT_CLIENT = Groq(api_key=GROQ_API_KEY)
        DEFAULT_CLIENT_ASYNC = AsyncGroq(api_key=GROQ_API_KEY)
        api_key = GROQ_API_KEY

class GroqChatArgs(BaseModelArgs):
    model: str = Field(default="mixtral-8x7b-32768")
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=1.0)
    top_p: int = Field(default=1)
    n: int = Field(default=1)
    stop: Optional[Union[str, List]] = Field(default=None)
    presence_penalty: int = Field(default=0)
    frequency_penalty: int = Field(default=0)

@llm_registry.register("groq")
class GroqChat(BaseChatModel):
    args: GroqChatArgs = Field(default_factory=GroqChatArgs)
    client_args: Optional[Dict] = Field(
        default={"api_key": api_key}
    )

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    def __init__(self, max_retry: int = 3, **kwargs):
        args = GroqChatArgs()
        args = args.dict()
        client_args = {"api_key": api_key}
        for k, v in args.items():
            args[k] = kwargs.pop(k, v)
        if len(kwargs) > 0:
            logger.warn(f"Unused arguments: {kwargs}")
        super().__init__(
            args=args, max_retry=max_retry, client_args=client_args
        )

    @classmethod
    def send_token_limit(self, model: str) -> int:
        send_token_limit_dict = {
            "llama3-8b-8192": 8192,
            "llama3-70b-8192": 8192,
            "mixtral-8x7b-32768": 32768,
            # Add other models as needed
        }
        # Default to 4096 tokens if model is not in the dictionary
        return send_token_limit_dict[model] if model in send_token_limit_dict else 4096

    # @retry(
    #     stop=stop_after_attempt(3),
    #     wait=wait_exponential(multiplier=1, min=4, max=10),
    #     reraise=True,
    # )
    async def generate_response(
        self,
        prepend_prompt: str = "",
        history: List[dict] = [],
        append_prompt: str = "",
        functions: List[dict] = [],
    ) -> LLMResult:
        messages = self.construct_messages(prepend_prompt, history, append_prompt)
        logger.log_prompt(messages)
        client = DEFAULT_CLIENT
        try:
            # Execute function call
            if functions != []:
                response = await client.chat.completions.create(
                    messages=messages,
                    functions=functions,
                    **self.args.dict(),
                )
                time.sleep(5)

                logger.log_prompt(
                    [
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    ]
                )
                if response.choices[0].message.function_call is not None:
                    self.collect_metrics(response)
                    return LLMResult(
                        content=response.choices[0].message.get("content", ""),
                        function_name=response.choices[0].message.function_call.name,
                        function_arguments=ast.literal_eval(
                            response.choices[0].message.function_call.arguments
                        ),
                        send_tokens=response.usage.prompt_tokens,
                        recv_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )
                else:
                    self.collect_metrics(response)
                    logger.log_prompt(
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    )
                    return LLMResult(
                        content=response.choices[0].message.content,
                        send_tokens=response.usage.prompt_tokens,
                        recv_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )
            else:
                response = await client.chat.completions.create(
                    messages=messages,
                    **self.args.dict(),
                )
                time.sleep(5)
                logger.log_prompt(
                    [
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    ]
                )
                self.collect_metrics(response)
                return LLMResult(
                    content=response.choices[0].message.content,
                    send_tokens=response.usage.prompt_tokens,
                    recv_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
        except (GroqError, KeyboardInterrupt, json.decoder.JSONDecodeError) as error:
            raise

    # @retry(
    #     stop=stop_after_attempt(3),
    #     wait=wait_exponential(multiplier=1, min=4, max=10),
    #     reraise=True,
    # )
    async def agenerate_response(
            self,
            prepend_prompt: str = "",
            history: List[dict] = [],
            append_prompt: str = "",
            functions: List[dict] = [],
    ) -> LLMResult:
        messages = self.construct_messages(prepend_prompt, history, append_prompt)
        logger.log_prompt(messages)

        async_client = DEFAULT_CLIENT
        try:
            if functions != []:
                response = async_client.chat.completions.create(
                    messages=messages,
                    functions=functions,
                    **self.args.dict(),
                )
                time.sleep(5)
                logger.log_prompt(
                    [
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    ]
                )
                if response.choices[0].message.function_call is not None:
                    function_name = response.choices[0].message.function_call.name
                    valid_function = False
                    if function_name.startswith("function."):
                        function_name = function_name.replace("function.", "")
                    elif function_name.startswith("functions."):
                        function_name = function_name.replace("functions.", "")
                    for function in functions:
                        if function["name"] == function_name:
                            valid_function = True
                            break
                    if not valid_function:
                        logger.warn(
                            f"The returned function name {function_name} is not in the list of valid functions. Retrying..."
                        )
                        raise ValueError(
                            f"The returned function name {function_name} is not in the list of valid functions."
                        )
                    try:
                        arguments = ast.literal_eval(
                            response.choices[0].message.function_call.arguments
                        )
                    except:
                        try:
                            arguments = ast.literal_eval(
                                JsonRepair(
                                    response.choices[0].message.function_call.arguments
                                ).repair()
                            )
                        except:
                            logger.warn(
                                "The returned argument in function call is not valid json. Retrying..."
                            )
                            raise ValueError(
                                "The returned argument in function call is not valid json."
                            )
                    self.collect_metrics(response)
                    logger.log_prompt(
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    )
                    return LLMResult(
                        function_name=function_name,
                        function_arguments=arguments,
                        send_tokens=response.usage.prompt_tokens,
                        recv_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )

                else:
                    self.collect_metrics(response)
                    logger.log_prompt(
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    )
                    return LLMResult(
                        content=response.choices[0].message.content,
                        send_tokens=response.usage.prompt_tokens,
                        recv_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )

            else:

                response = async_client.chat.completions.create(
                    messages=messages,
                    **self.args.dict(),
                )
                time.sleep(5)
                self.collect_metrics(response)
                logger.log_prompt(
                    [
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    ]
                )
                return LLMResult(
                    content=response.choices[0].message.content,
                    send_tokens=response.usage.prompt_tokens,
                    recv_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
        except (GroqError, KeyboardInterrupt, json.decoder.JSONDecodeError) as error:
            raise

    def construct_messages(
        self, prepend_prompt: str, history: List[dict], append_prompt: str
    ):
        messages = []
        if prepend_prompt != "":
            messages.append({"role": "system", "content": prepend_prompt})
        if len(history) > 0:
            messages += history
        if append_prompt != "":
            messages.append({"role": "user", "content": append_prompt})
        return messages

    def collect_metrics(self, response):
        self.total_prompt_tokens += response.usage.prompt_tokens
        self.total_completion_tokens += response.usage.completion_tokens

    def get_spend(self) -> int:
        input_cost_map = {
            "llama3-8b-8192": 0,
            "llama3-70b-8192": 0,
            "mixtral-8x7b-32768": 0,
            # Add other models as needed
        }

        output_cost_map = {
            "llama3-8b-8192": 0,
            "llama3-70b-8192": 0,
            "mixtral-8x7b-32768": 0,
            # Add other models as needed
        }

        model = self.args.model
        if model not in input_cost_map or model not in output_cost_map:
            raise ValueError(f"Model type {model} not supported")

        return (
            self.total_prompt_tokens * input_cost_map[model] / 1000.0
            + self.total_completion_tokens * output_cost_map[model] / 1000.0
        )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def get_embedding(text: str, attempts=3) -> np.array:
    client = Groq(api_key=GROQ_API_KEY)
    try:
        text = text.replace("\n", " ")
        embedding = client.create_embedding(
            input=text, model="groq-embedding-001"
        ).model_dump_json(indent=2)
        return tuple(embedding)
    except Exception as e:
        attempt += 1
        logger.error(f"Error {e} when requesting groq models. Retrying")
        raise
