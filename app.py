import streamlit as st
import json
from typing import Iterable, Optional
from moa.agent import MOAgent
from moa.agent.moa import ResponseChunk
from streamlit_ace import st_ace
import copy
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# Update the default configuration
default_config = {
    "main_model": "llama3.1:8b-instruct-q6_K",
    "main_system_prompt": "You are a helpful assistant. Written text should always use British English spelling.",
    "cycles": 2,
    "layer_agent_config": {},
}

layer_agent_config_def = {
    "layer_agent_1": {
        "system_prompt": "Written text should always use British English spelling. Think through your response step by step. {helper_response}",
        "model_name": "llama3.1:8b-instruct-q6_K",
        "temperature": 0.7,
    },
    "layer_agent_2": {
        "system_prompt": "Written text should always use British English spelling. Respond with a thought and then your response to the question. {helper_response}",
        "model_name": "qwen2-7b-maziyarpanahi-v0_8-instruct:Q6_K",
        "temperature": 0.5,
    },
    "layer_agent_3": {
        "system_prompt": "You are an expert programmer. Written text should always use British English spelling. Always use the latest libraries and techniques. {helper_response}",
        "model_name": "mistral-nemo:12b-instruct-2407-q6_K",
        "temperature": 0.3,
    },
}

valid_model_names = [
    "llama3.1:8b-instruct-q6_K",
    "qwen2-7b-maziyarpanahi-v0_8-instruct:Q6_K",
    "mistral-nemo:12b-instruct-2407-q6_K",
    "deepseek-coder-v2-lite-instruct:q6_k_l",
    "codestral-22b_ef16:q6_k",
    "llama3.1:70b-instruct-q4_K_M",
    "qwen2-72b-maziyarpanahi-v0_1-instruct:IQ4_XS",
    "mistral-large-instruct-2407:iq2_m",
]


def add_logo():
    logo = Image.open("static/logo.png")
    st.sidebar.image(logo, width=150)


def api_request_callback(request):
    if st.session_state.log_api_requests:
        st.write(f"API Request: {request}")


def update_layer_config(cycles):
    current_config = st.session_state.layer_agent_config
    new_config = {}
    for i in range(1, cycles + 1):
        layer_key = f"layer_agent_{i}"
        if layer_key in current_config:
            new_config[layer_key] = current_config[layer_key]
        else:
            new_config[layer_key] = copy.deepcopy(
                layer_agent_config_def["layer_agent_1"]
            )
    st.session_state.layer_agent_config = new_config


def stream_response(messages: Iterable[ResponseChunk]):
    layer_outputs = {}
    for message in messages:
        if message["response_type"] == "intermediate":
            layer = message["metadata"]["layer"]
            if layer not in layer_outputs:
                layer_outputs[layer] = []
            layer_outputs[layer].append(message["delta"])
        else:
            if layer_outputs:
                for layer, outputs in layer_outputs.items():
                    st.write(f"Layer {layer}")
                    cols = st.columns(len(outputs))
                    for i, output in enumerate(outputs):
                        with cols[i]:
                            st.expander(label=f"Agent {i+1}", expanded=False).write(
                                output
                            )
                layer_outputs = {}
            yield message["delta"]


def set_moa_agent(**kwargs):
    for key, value in kwargs.items():
        if value is not None:
            setattr(st.session_state, key, value)

    main_model_kwargs = {
        "temperature": st.session_state.main_temperature,
        "max_tokens": st.session_state.main_max_tokens,
    }

    optional_params = [
        ("top_p", "main_top_p"),
        ("top_k", "main_top_k"),
        ("min_p", "main_min_p"),
        ("repetition_penalty", "main_repetition_penalty"),
        ("presence_penalty", "main_presence_penalty"),
        ("frequency_penalty", "main_frequency_penalty"),
    ]

    for api_param, state_param in optional_params:
        value = getattr(st.session_state, state_param)
        if value is not None and value != 0:  # Assume 0 means disabled for these params
            main_model_kwargs[api_param] = value

    if st.session_state.main_api_base:
        main_model_kwargs["api_base"] = st.session_state.main_api_base
    if st.session_state.main_api_key:
        main_model_kwargs["api_key"] = st.session_state.main_api_key

    if st.session_state.main_num_ctx > 0:
        main_model_kwargs["num_ctx"] = st.session_state.main_num_ctx
    if st.session_state.main_num_batch > 0:
        main_model_kwargs["num_batch"] = st.session_state.main_num_batch

    st.session_state.moa_agent = MOAgent.from_config(
        main_model=st.session_state.main_model,
        main_system_prompt=st.session_state.main_system_prompt,
        cycles=st.session_state.cycles,
        layer_agent_config=copy.deepcopy(st.session_state.layer_agent_config),
        **main_model_kwargs,
    )


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    default_values = {
        "main_model": "llama3.1:8b-instruct-q6_K",
        "main_system_prompt": "You are a helpful assistant. Written text should always use British English spelling.",
        "cycles": 2,
        "layer_agent_config": copy.deepcopy(layer_agent_config_def),
        "main_temperature": 0.7,
        "main_max_tokens": 2048,
        "main_api_base": "",
        "main_api_key": "",
        "main_num_ctx": 2048,
        "log_api_requests": False,
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            setattr(st.session_state, key, value)

    # Initialize optional parameters as None
    optional_params = [
        "main_top_p",
        "main_top_k",
        "main_min_p",
        "main_repetition_penalty",
        "main_presence_penalty",
        "main_frequency_penalty",
        "main_num_batch",
    ]
    for param in optional_params:
        if param not in st.session_state:
            setattr(st.session_state, param, None)


def render_sidebar():
    with st.sidebar:
        st.title("Mixture of (Ollama) Agents")

        with st.expander("Main Model Settings", expanded=False):
            st.session_state.main_model = st.selectbox(
                "Select Main Model",
                options=valid_model_names,
                index=valid_model_names.index(st.session_state.main_model),
            )
            new_cycles = st.number_input(
                "Number of Layers",
                min_value=1,
                max_value=10,
                value=st.session_state.cycles,
                key="cycles_input",
            )
            if new_cycles != st.session_state.cycles:
                st.session_state.cycles = new_cycles
                update_layer_config(new_cycles)
                st.rerun()

            st.session_state.main_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.main_temperature,
                step=0.1,
            )
            st.session_state.main_max_tokens = st.number_input(
                "Max Tokens",
                min_value=1,
                max_value=8192,
                value=st.session_state.main_max_tokens,
            )

        with st.expander("Layer Agent Configuration", expanded=False):
            for i in range(1, st.session_state.cycles + 1):
                st.subheader(f"Layer Agent {i}")
                layer_key = f"layer_agent_{i}"

                st.session_state.layer_agent_config[layer_key]["model_name"] = (
                    st.selectbox(
                        f"Model for Layer {i}",
                        options=valid_model_names,
                        index=valid_model_names.index(
                            st.session_state.layer_agent_config[layer_key]["model_name"]
                        ),
                        key=f"layer_model_{i}",
                    )
                )
                st.session_state.layer_agent_config[layer_key]["system_prompt"] = (
                    st.text_area(
                        f"System Prompt for Layer {i}",
                        value=st.session_state.layer_agent_config[layer_key][
                            "system_prompt"
                        ],
                        key=f"layer_prompt_{i}",
                    )
                )
                st.session_state.layer_agent_config[layer_key]["temperature"] = (
                    st.slider(
                        f"Temperature for Layer {i}",
                        min_value=0.0,
                        max_value=2.0,
                        value=st.session_state.layer_agent_config[layer_key][
                            "temperature"
                        ],
                        step=0.1,
                        key=f"layer_temp_{i}",
                    )
                )

        with st.expander("Advanced Settings", expanded=False):
            st.session_state.main_top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=(
                    st.session_state.main_top_p
                    if st.session_state.main_top_p is not None
                    else 1.0
                ),
                step=0.01,
                help="Set to None to disable",
            )
            st.session_state.main_top_k = st.number_input(
                "Top K",
                min_value=1,
                max_value=100,
                value=(
                    st.session_state.main_top_k
                    if st.session_state.main_top_k is not None
                    else 40
                ),
                help="Set to 0 to disable",
            )
            st.session_state.main_min_p = st.slider(
                "Min P",
                min_value=0.0,
                max_value=1.0,
                value=(
                    st.session_state.main_min_p
                    if st.session_state.main_min_p is not None
                    else 0.05
                ),
                step=0.01,
                help="Set to 0 to disable",
            )
            st.session_state.main_repetition_penalty = st.slider(
                "Repetition Penalty",
                min_value=1.0,
                max_value=2.0,
                value=(
                    st.session_state.main_repetition_penalty
                    if st.session_state.main_repetition_penalty is not None
                    else 1.1
                ),
                step=0.01,
                help="Set to 1 to disable",
            )
            st.session_state.main_presence_penalty = st.slider(
                "Presence Penalty",
                min_value=0.0,
                max_value=2.0,
                value=(
                    st.session_state.main_presence_penalty
                    if st.session_state.main_presence_penalty is not None
                    else 0.0
                ),
                step=0.01,
                help="Set to 0 to disable",
            )
            st.session_state.main_frequency_penalty = st.slider(
                "Frequency Penalty",
                min_value=0.0,
                max_value=2.0,
                value=(
                    st.session_state.main_frequency_penalty
                    if st.session_state.main_frequency_penalty is not None
                    else 0.0
                ),
                step=0.01,
                help="Set to 0 to disable",
            )
        with st.expander("API Settings", expanded=False):
            st.session_state.main_api_base = st.text_input(
                "API Base URL", value=st.session_state.main_api_base
            )
            st.session_state.main_api_key = st.text_input(
                "API Key", value=st.session_state.main_api_key, type="password"
            )
            st.session_state.main_num_ctx = st.number_input(
                "Context Size (num_ctx)",
                min_value=1,
                max_value=32768,
                value=st.session_state.main_num_ctx,
                help="Number of context tokens. Set to 0 to use model default.",
            )
            st.session_state.main_num_batch = st.number_input(
                "Batch Size (num_batch)",
                min_value=0,
                max_value=4096,
                value=(
                    st.session_state.main_num_batch
                    if st.session_state.main_num_batch is not None
                    else 0
                ),
                help="Batch size. Set to 0 to use model default.",
            )

        with st.expander("System Prompt", expanded=False):
            main_system_prompt = st.text_area(
                "Main System Prompt",
                value=st.session_state.main_system_prompt,
                height=100,
            )

        log_api_requests = st.checkbox("Log API Requests", value=False)

        if st.button("Update Configuration"):
            try:
                set_moa_agent(
                    main_model=new_main_model,
                    main_system_prompt=main_system_prompt,
                    cycles=new_cycles,
                    layer_agent_config=st.session_state.layer_agent_config,
                    main_model_temperature=main_temperature,
                    main_model_max_tokens=main_max_tokens,
                    main_model_top_p=main_top_p,
                    main_model_top_k=main_top_k,
                    main_model_min_p=main_min_p,
                    main_model_repetition_penalty=main_repetition_penalty,
                    main_model_presence_penalty=main_presence_penalty,
                    main_model_frequency_penalty=main_frequency_penalty,
                    main_model_api_base=main_api_base,
                    main_model_api_key=main_api_key,
                    main_model_num_ctx=main_num_ctx,
                    main_model_num_batch=main_num_batch,
                )
                st.session_state.messages = []
                st.session_state.log_api_requests = log_api_requests
                st.success("Configuration updated successfully!")
            except Exception as e:
                st.error(f"Error updating configuration: {str(e)}")

        if st.button("Use Recommended Config"):
            try:
                recommended_config = {
                    "main_model": "llama3.1:8b-instruct-q6_K",
                    "cycles": 2,
                    "layer_agent_config": {
                        "layer_agent_1": {
                            "system_prompt": "Written text should always use British English spelling. Think through your response step by step. {helper_response}",
                            "model_name": "llama3.1:8b-instruct-q6_K",
                            "temperature": 0.7,
                        },
                        "layer_agent_2": {
                            "system_prompt": "Written text should always use British English spelling. Respond with a thought and then your response to the question. {helper_response}",
                            "model_name": "qwen2-7b-maziyarpanahi-v0_8-instruct:Q6_K",
                            "temperature": 0.5,
                        },
                    },
                }
                set_moa_agent(**recommended_config)
                st.session_state.messages = []
                st.success("Configuration updated to recommended settings!")
            except Exception as e:
                st.error(f"Error updating to recommended configuration: {str(e)}")

        # Add current configuration display
        with st.expander("Current MOA Configuration", expanded=False):
            st.markdown(f"**Main Model**: `{st.session_state.main_model}`")
            st.markdown(f"**Layers**: `{st.session_state.cycles}`")
            st.markdown(
                f"**Main System Prompt**: `{st.session_state.main_system_prompt}`"
            )
            st.markdown(
                f"**Main Model Temperature**: `{st.session_state.main_temperature:.1f}`"
            )
            st.markdown(
                f"**Main Model Max Tokens**: `{st.session_state.main_max_tokens}`"
            )
            st.markdown(f"**Layer Agents Config**:")
            st.json(st.session_state.layer_agent_config)


def render_chat_interface():
    st.markdown(
        "<h4 style='text-align: left; font-size: 20px;'>Mixture Of Agents</h4>",
        unsafe_allow_html=True,
    )
    st.image("./static/banner.png", caption="", width=200)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        moa_agent: MOAgent = st.session_state.moa_agent
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            ast_mess = stream_response(moa_agent.chat(query, output_format="json"))
            response = st.write_stream(ast_mess)

        st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    st.set_page_config(
        page_title="Mixture-Of-Agents",
        page_icon="static/favicon.ico",
        menu_items={"About": "## Ollama Mixture-Of-Agents"},
        layout="wide",
    )
    add_logo()
    initialize_session_state()
    render_sidebar()
    set_moa_agent()
    render_chat_interface()


if __name__ == "__main__":
    main()
