import streamlit as st
import json
from typing import Iterable, Optional
from moa.agent import MOAgent
from moa.agent.moa import ResponseChunk
from streamlit_ace import st_ace
import copy
from dotenv import load_dotenv
import streamlit as st
from PIL import Image

# import sys
# print(sys.path)

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

# Recommended Configuration
rec_config = {
    "main_model": "llama3.1:8b-instruct-q6_K",
    "cycles": 2,
    "layer_agent_config": {},
}

layer_agent_config_rec = {
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
    "layer_agent_4": {
        "system_prompt": "You are an expert programmer. Written text should always use British English spelling. Always use the latest libraries and techniques. {helper_response}",
        "model_name": "deepseek-coder-v2-lite-instruct:q6_k_l",
        "temperature": 0.3,
    },
}


def add_logo():
    logo = Image.open("static/logo.png")
    st.sidebar.image(logo, width=150)  # Adjust width as needed


def api_request_callback(request):
    if st.session_state.log_api_requests:
        st.write(f"API Request: {request}")


def stream_response(messages: Iterable[ResponseChunk]):
    layer_outputs = {}
    for message in messages:
        if message["response_type"] == "intermediate":
            layer = message["metadata"]["layer"]
            if layer not in layer_outputs:
                layer_outputs[layer] = []
            layer_outputs[layer].append(message["delta"])
        else:
            # Display accumulated layer outputs
            for layer, outputs in layer_outputs.items():
                st.write(f"Layer {layer}")
                cols = st.columns(len(outputs))
                for i, output in enumerate(outputs):
                    with cols[i]:
                        st.expander(label=f"Agent {i+1}", expanded=False).write(output)

            # Clear layer outputs for the next iteration
            layer_outputs = {}

            # Yield the main agent's output
            yield message["delta"]


def set_moa_agent(
    main_model: str = default_config["main_model"],
    main_system_prompt: str = default_config["main_system_prompt"],
    cycles: int = default_config["cycles"],
    layer_agent_config: dict[dict[str, any]] = copy.deepcopy(layer_agent_config_def),
    main_model_temperature: Optional[float] = None,
    main_model_max_tokens: Optional[int] = 2048,
    main_model_top_p: Optional[float] = 0.9,
    main_model_top_k: Optional[int] = None,
    main_model_min_p: Optional[float] = None,
    main_model_repetition_penalty: Optional[float] = None,
    main_model_presence_penalty: Optional[float] = None,
    main_model_frequency_penalty: Optional[float] = None,
    main_model_api_base: Optional[str] = None,
    main_model_api_key: Optional[str] = None,
    main_model_num_ctx: Optional[int] = 2048,  # None,
    main_model_num_batch: Optional[int] = None,
    override: bool = False,
):
    # Initialize session state variables with default values if they don't exist
    if "main_model" not in st.session_state:
        st.session_state.main_model = main_model
    if "main_system_prompt" not in st.session_state:
        st.session_state.main_system_prompt = main_system_prompt
    if "cycles" not in st.session_state:
        st.session_state.cycles = cycles
    if "layer_agent_config" not in st.session_state:
        st.session_state.layer_agent_config = layer_agent_config
    if "main_temperature" not in st.session_state:
        st.session_state.main_temperature = (
            main_model_temperature or 0.7
        )  # Default value
    if "main_max_tokens" not in st.session_state:
        st.session_state.main_max_tokens = main_model_max_tokens  # can be None
    if "main_top_p" not in st.session_state:
        st.session_state.main_top_p = main_model_top_p  # Can be None
    if "main_top_k" not in st.session_state:
        st.session_state.main_top_k = main_model_top_k  # Can be None
    if "main_min_p" not in st.session_state:
        st.session_state.main_min_p = main_model_min_p  # Can be None
    if "main_repetition_penalty" not in st.session_state:
        st.session_state.main_repetition_penalty = (
            main_model_repetition_penalty  # Can be None  # Default value
        )
    if "main_presence_penalty" not in st.session_state:
        st.session_state.main_presence_penalty = (
            main_model_presence_penalty  # Can be None  # Default value
        )
    if "main_frequency_penalty" not in st.session_state:
        st.session_state.main_frequency_penalty = (
            main_model_frequency_penalty  # Can be None  # Default value
        )
    if "main_api_base" not in st.session_state:
        st.session_state.main_api_base = main_model_api_base or ""  # Default value
    if "main_api_key" not in st.session_state:
        st.session_state.main_api_key = main_model_api_key or ""  # Default value
    if "main_num_ctx" not in st.session_state:
        st.session_state.main_num_ctx = main_model_num_ctx  # Can be None
    if "main_num_batch" not in st.session_state:
        st.session_state.main_num_batch = main_model_num_batch  # Can be None

    # Update session state variables only if override is True or they don't exist
    if override or "main_model" not in st.session_state:
        st.session_state.main_model = main_model
    if override or "main_system_prompt" not in st.session_state:
        st.session_state.main_system_prompt = main_system_prompt
    if override or "cycles" not in st.session_state:
        st.session_state.cycles = cycles
    if override or "layer_agent_config" not in st.session_state:
        st.session_state.layer_agent_config = layer_agent_config

    # Update optional parameters only if they are provided
    if main_model_temperature is not None:
        st.session_state.main_temperature = main_model_temperature
    if main_model_max_tokens is not None:
        st.session_state.main_max_tokens = main_model_max_tokens
    if main_model_top_p is not None:
        st.session_state.main_top_p = main_model_top_p
    if main_model_top_k is not None:
        st.session_state.main_top_k = main_model_top_k
    if main_model_min_p is not None:
        st.session_state.main_min_p = main_model_min_p
    if main_model_repetition_penalty is not None:
        st.session_state.main_repetition_penalty = main_model_repetition_penalty
    if main_model_presence_penalty is not None:
        st.session_state.main_presence_penalty = main_model_presence_penalty
    if main_model_frequency_penalty is not None:
        st.session_state.main_frequency_penalty = main_model_frequency_penalty
    if main_model_api_base is not None:
        st.session_state.main_api_base = main_model_api_base
    if main_model_api_key is not None:
        st.session_state.main_api_key = main_model_api_key
    if main_model_num_ctx is not None:
        st.session_state.main_num_ctx = main_model_num_ctx
    if main_model_num_batch is not None:
        st.session_state.main_num_batch = main_model_num_batch

    main_model_kwargs = {}
    optional_params = [
        ("temperature", st.session_state.main_temperature),
        ("max_tokens", st.session_state.main_max_tokens),
        ("top_p", st.session_state.main_top_p),
        ("top_k", st.session_state.main_top_k),
        ("min_p", st.session_state.main_min_p),
        ("repetition_penalty", st.session_state.main_repetition_penalty),
        ("presence_penalty", st.session_state.main_presence_penalty),
        ("frequency_penalty", st.session_state.main_frequency_penalty),
    ]

    for param_name, param_value in optional_params:
        if param_value is not None:
            main_model_kwargs[param_name] = param_value

    if st.session_state.main_api_base:
        main_model_kwargs["api_base"] = st.session_state.main_api_base
    if st.session_state.main_api_key:
        main_model_kwargs["api_key"] = st.session_state.main_api_key

    model_kwargs = {}
    if st.session_state.main_num_ctx is not None:
        model_kwargs["num_ctx"] = st.session_state.main_num_ctx
    if st.session_state.main_num_batch is not None:
        model_kwargs["num_batch"] = st.session_state.main_num_batch

    if model_kwargs:
        main_model_kwargs["model_kwargs"] = model_kwargs

    st.session_state.moa_agent = MOAgent.from_config(
        main_model=st.session_state.main_model,
        main_system_prompt=st.session_state.main_system_prompt,
        cycles=st.session_state.cycles,
        layer_agent_config=copy.deepcopy(st.session_state.layer_agent_config),
        **main_model_kwargs,
    )


st.set_page_config(
    page_title="Mixture-Of-Agents",
    page_icon="static/favicon.ico",
    menu_items={"About": "## Ollama Mixture-Of-Agents"},
    layout="wide",
)
add_logo()

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


# st.markdown("<h1>Mixture-Of-Agents</h1>", unsafe_allow_html=True)
# st.write("---")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

set_moa_agent()

# Sidebar for configuration
with st.sidebar:
    st.title("Mixture of (Ollama) Agents")
    with st.expander("Main Model Settings", expanded=False):
        new_main_model = st.selectbox(
            "Select Main Model",
            options=valid_model_names,
            index=valid_model_names.index(st.session_state.main_model),
        )
        new_cycles = st.number_input(
            "Number of Layers", min_value=1, max_value=10, value=st.session_state.cycles
        )
        main_temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.main_temperature,
            step=0.1,
        )
        main_max_tokens = st.number_input(
            "Max Tokens",
            min_value=1,
            max_value=8192,
            value=st.session_state.main_max_tokens,
        )

    with st.expander("Advanced Settings", expanded=False):
        main_top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.main_top_p,
            step=0.1,
        )
        main_top_k = st.number_input(
            "Top K", min_value=1, max_value=100, value=st.session_state.main_top_k
        )
        main_min_p = st.slider(
            "Min P",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.main_min_p,
            step=0.01,
        )
        main_repetition_penalty = st.slider(
            "Repetition Penalty",
            min_value=1.0,
            max_value=2.0,
            value=st.session_state.main_repetition_penalty,
            step=0.1,
        )
        main_presence_penalty = st.slider(
            "Presence Penalty",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.main_presence_penalty,
            step=0.1,
        )
        main_frequency_penalty = st.slider(
            "Frequency Penalty",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.main_frequency_penalty,
            step=0.1,
        )

    with st.expander("API Settings", expanded=False):
        main_api_base = st.text_input(
            "API Base URL", value=st.session_state.main_api_base
        )
        main_api_key = st.text_input(
            "API Key", value=st.session_state.main_api_key, type="password"
        )
        main_num_ctx = st.number_input(
            "Context Size (num_ctx)",
            min_value=1,
            max_value=32768,
            value=st.session_state.main_num_ctx,
            help="Number of context tokens. Leave empty to use model default.",
        )
        main_num_batch = st.number_input(
            "Batch Size (num_batch)",
            min_value=1,
            max_value=4096,
            value=st.session_state.main_num_batch,
            help="Batch size. Leave empty to use model default.",
        )

    with st.expander("System Prompt", expanded=False):
        main_system_prompt = st.text_area(
            "Main System Prompt",
            value=st.session_state.main_system_prompt,
            height=100,
        )

    with st.expander("Layer Agent Config", expanded=False):
        new_layer_agent_config = st_ace(
            value=json.dumps(st.session_state.layer_agent_config, indent=2),
            language="json",
            theme="github",
            keybinding="vscode",
            show_gutter=True,
            min_lines=10,
            max_lines=30,
            font_size=14,
            wrap=True,
            auto_update=True,
        )

    log_api_requests = st.checkbox("Log API Requests", value=False)

    if st.button("Update Configuration"):
        try:
            new_layer_config = json.loads(new_layer_agent_config)
            set_moa_agent(
                main_model=new_main_model,
                main_system_prompt=main_system_prompt,
                cycles=new_cycles,
                layer_agent_config=new_layer_config,
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
                override=True,
            )
            st.session_state.messages = []
            st.session_state.log_api_requests = log_api_requests
            st.success("Configuration updated successfully!")
        except json.JSONDecodeError:
            st.error(
                "Invalid JSON in Layer Agent Configuration. Please check your input."
            )
        except Exception as e:
            st.error(f"Error updating configuration: {str(e)}")

    if st.button("Use Recommended Config"):
        try:
            set_moa_agent(
                main_model=rec_config["main_model"],
                cycles=rec_config["cycles"],
                layer_agent_config=layer_agent_config_rec,
                override=True,
            )
            st.session_state.messages = []
            st.success("Configuration updated to recommended settings!")
        except Exception as e:
            st.error(f"Error updating to recommended configuration: {str(e)}")

# Main app layout
st.markdown(
    "<h4 style='text-align: left; font-size: 20px;'>Mixture Of Agents</h4>",
    unsafe_allow_html=True,
)
st.image("./static/banner.png", caption="", width=200)


# Chat interface
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

# Display current configuration
with st.sidebar:
    with st.expander("Current MOA Configuration", expanded=False):
        st.markdown(f"**Main Model**: ``{st.session_state.main_model}``")
        st.markdown(
            f"**Main Model Temperature**: ``{st.session_state.main_temperature:.1f}``"
        )
        st.markdown(
            f"**Main Model num_ctx**: ``{st.session_state.main_num_ctx or 'Default'}``"
        )
        st.markdown(
            f"**Main Model num_batch**: ``{st.session_state.main_num_batch or 'Default'}``"
        )
        st.markdown(
            f"**Main Model Max Tokens**: ``{st.session_state.main_max_tokens}``"
        )
        st.markdown(f"**Main Model Top P**: ``{st.session_state.main_top_p:.2f}``")
        st.markdown(f"**Main Model Top K**: ``{st.session_state.main_top_k}``")
        st.markdown(f"**Main Model Min P**: ``{st.session_state.main_min_p:.2f}``")
        st.markdown(
            f"**Main Model Repetition Penalty**: ``{st.session_state.main_repetition_penalty:.2f}``"
        )
        st.markdown(
            f"**Main Model Presence Penalty**: ``{st.session_state.main_presence_penalty:.2f}``"
        )
        st.markdown(
            f"**Main Model Frequency Penalty**: ``{st.session_state.main_frequency_penalty:.2f}``"
        )
        st.markdown(f"**Main Model API Base**: ``{st.session_state.main_api_base}``")
        st.markdown(
            f"**Main Model API Key**: ``{'*' * len(st.session_state.main_api_key) if st.session_state.main_api_key else 'Not set'}``"
        )
        st.markdown(f"**Layers**: ``{st.session_state.cycles}``")
        st.markdown(f"**Layer Agents Config**:")
        new_layer_agent_config = st_ace(
            value=json.dumps(st.session_state.layer_agent_config, indent=2),
            language="json",
            placeholder="Layer Agent Configuration (JSON)",
            show_gutter=False,
            wrap=True,
            readonly=True,
            auto_update=True,
        )
