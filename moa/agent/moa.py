"""
Ollama API compatible agent
"""

import os
from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any, Callable
from dotenv import load_dotenv

import streamlit as st

from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableSerializable,
)
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks import StdOutCallbackHandler
from .prompts import DEFAULT_SYSTEM_PROMPT, REFERENCE_SYSTEM_PROMPT

load_dotenv()


class ResponseChunk(TypedDict):
    delta: str
    response_type: Literal["intermediate", "output"]
    metadata: Dict[str, Any]


class MOAgent:
    def __init__(
        self,
        main_agent: RunnableSerializable[Dict, str],
        layer_agent: RunnableSerializable[Dict, Dict],
        reference_system_prompt: Optional[str] = None,
        cycles: Optional[int] = None,
        chat_memory: Optional[ConversationBufferMemory] = None,
        api_request_callback: Optional[Callable[[Dict], None]] = None,
    ) -> None:
        self.reference_system_prompt = (
            reference_system_prompt or REFERENCE_SYSTEM_PROMPT
        )
        self.main_agent = main_agent
        self.layer_agent = layer_agent
        self.cycles = cycles or 1
        self.chat_memory = chat_memory or ConversationBufferMemory(
            memory_key="messages", return_messages=True
        )
        self.api_request_callback = api_request_callback

    @staticmethod
    def concat_response(
        inputs: Dict[str, str], reference_system_prompt: Optional[str] = None
    ):
        reference_system_prompt = reference_system_prompt or REFERENCE_SYSTEM_PROMPT

        responses = ""
        res_list = []
        for i, out in enumerate(inputs.values()):
            responses += f"{i}. {out}\n"
            res_list.append(out)

        formatted_prompt = reference_system_prompt.format(responses=responses)
        return {"formatted_response": formatted_prompt, "responses": res_list}

    @classmethod
    def from_config(
        cls,
        main_model: Optional[str] = "rys-llama3.1:8b-instruct-Q8_0",
        main_system_prompt: Optional[str] = None,
        cycles: int = 1,
        layer_agent_config: Optional[Dict] = None,
        reference_system_prompt: Optional[str] = None,
        api_request_callback: Optional[Callable[[Dict], None]] = None,
        **main_model_kwargs,
    ):
        reference_system_prompt = reference_system_prompt or REFERENCE_SYSTEM_PROMPT
        main_system_prompt = main_system_prompt or DEFAULT_SYSTEM_PROMPT
        layer_agent = MOAgent._configure_layer_agent(
            layer_agent_config, api_request_callback
        )
        main_agent = MOAgent._create_agent_from_system_prompt(
            system_prompt=main_system_prompt,
            model_name=main_model,
            api_request_callback=api_request_callback,
            **main_model_kwargs,
        )
        return cls(
            main_agent=main_agent,
            layer_agent=layer_agent,
            reference_system_prompt=reference_system_prompt,
            cycles=cycles,
            api_request_callback=api_request_callback,
        )

    @staticmethod
    def _configure_layer_agent(
        layer_agent_config: Optional[Dict] = None,
        api_request_callback: Optional[Callable[[Dict], None]] = None,
    ) -> RunnableSerializable[Dict, Dict]:
        if not layer_agent_config:
            layer_agent_config = {
                "layer_agent_1": {
                    "system_prompt": DEFAULT_SYSTEM_PROMPT,
                    "model_name": "rys-llama3.1:8b-instruct-Q8_0",
                },
                "layer_agent_2": {
                    "system_prompt": DEFAULT_SYSTEM_PROMPT,
                    "model_name": "qwen2:7b-instruct-q6_K",
                },
                "layer_agent_3": {
                    "system_prompt": DEFAULT_SYSTEM_PROMPT,
                    "model_name": "mistral-nemo:12b-instruct-2407-q6_K",
                },
            }

        parallel_chain_map = dict()
        for key, value in layer_agent_config.items():
            chain = MOAgent._create_agent_from_system_prompt(
                system_prompt=value.pop("system_prompt", DEFAULT_SYSTEM_PROMPT),
                model_name=value.pop("model_name", "rys-llama3.1:8b-instruct-Q8_0"),
                api_request_callback=api_request_callback,
                **value,
            )
            parallel_chain_map[key] = RunnablePassthrough() | chain

        chain = parallel_chain_map | RunnableLambda(MOAgent.concat_response)
        return chain

    @staticmethod
    def _create_agent_from_system_prompt(
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        model_name: str = "rys-llama3.1:8b-instruct-Q8_0",
        api_request_callback: Optional[Callable[[Dict], None]] = None,
        **llm_kwargs,
    ) -> RunnableSerializable[Dict, str]:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages", optional=True),
                ("human", "{input}"),
                ("system", "{helper_response}"),
            ]
        )

        callbacks = []
        if api_request_callback:
            class CustomCallback(StdOutCallbackHandler):
                def on_llm_start(self, serialized, prompts, **kwargs):
                    api_request_callback({"type": "llm_start", "prompts": prompts})

                def on_llm_end(self, response, **kwargs):
                    api_request_callback(
                        {"type": "llm_end", "response": response.generations[0][0].text}
                    )

            callbacks.append(CustomCallback())

        # Extract num_ctx and num_batch
        num_ctx = llm_kwargs.pop("num_ctx", None)
        num_batch = llm_kwargs.pop("num_batch", None)

        # Create the Ollama instance with the correct parameters
        ollama_kwargs = {
            "model": model_name,
            "callbacks": callbacks,
        }
        if num_ctx is not None:
            ollama_kwargs["num_ctx"] = int(num_ctx)
        if num_batch is not None:
            ollama_kwargs["num_batch"] = int(num_batch)
        ollama_kwargs.update(llm_kwargs)

        # set the base url for the Ollama instance
        if "base_url" not in ollama_kwargs:
            ollama_kwargs["base_url"] = os.getenv(
                "OLLAMA_HOST",
            )

        llm = Ollama(**ollama_kwargs)

        chain = prompt | llm | StrOutputParser()
        return chain

    def chat(
        self,
        input: str,
        messages: Optional[List[BaseMessage]] = None,
        cycles: Optional[int] = None,
        save: bool = True,
        output_format: Literal["string", "json"] = "string",
        api_request_callback: Optional[Callable[[Dict], None]] = None,
    ) -> Generator[str | ResponseChunk, None, None]:
        cycles = cycles or self.cycles
        llm_inp = {
            "input": input,
            "messages": messages
            or self.chat_memory.load_memory_variables({})["messages"],
            "helper_response": "",
        }
        for cyc in range(cycles):
            layer_output = self.layer_agent.invoke(llm_inp)
            l_frm_resp = layer_output["formatted_response"]
            l_resps = layer_output["responses"]

            llm_inp = {
                "input": input,
                "messages": self.chat_memory.load_memory_variables({})["messages"],
                "helper_response": l_frm_resp,
            }

            if output_format == "json":
                for l_out in l_resps:
                    yield ResponseChunk(
                        delta=l_out,
                        response_type="intermediate",
                        metadata={"layer": cyc + 1},
                    )

        stream = self.main_agent.stream(llm_inp)
        response = ""
        for chunk in stream:
            if output_format == "json":
                yield ResponseChunk(delta=chunk, response_type="output", metadata={})
            else:
                yield chunk
            response += chunk

        if save:
            self.chat_memory.save_context({"input": input}, {"output": response})

        if api_request_callback:
            api_request_callback(
                {"type": "chat_complete", "input": input, "output": response}
            )


def set_custom_font_size(base_font_size: int = 16):
    custom_css = f"""
    <style>
        html, body, [class*="css"] {{
            font-size: {base_font_size}px !important;
        }}
        .stTextInput > div > div > input {{
            font-size: {base_font_size}px;
        }}
        .stTextArea > div > div > textarea {{
            font-size: {base_font_size}px;
        }}
        .stSelectbox > div > div > div {{
            font-size: {base_font_size}px;
        }}
        .stMarkdown {{
            font-size: {base_font_size}px;
        }}
        .stChat {{
            font-size: {base_font_size}px;
        }}
        /* You can add more specific selectors here if needed */
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
