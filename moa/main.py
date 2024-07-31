from agent import MOAgent

# Configure agent
layer_agent_config = {
    "layer_agent_1": {
        "system_prompt": "Think through your response with step by step {helper_response}",
        "model_name": "llama3.1:8b-instruct-q6_K",
    },
    "layer_agent_2": {
        "system_prompt": "Respond with a thought and then your response to the question {helper_response}",
        "model_name": "qwen2-7b-maziyarpanahi-v0_8-instruct:Q6_K",
    },
    "layer_agent_3": {"model_name": "llama3.1:8b-instruct-q6_K"},
    "layer_agent_4": {"model_name": "qwen2-7b-maziyarpanahi-v0_8-instruct:Q6_K"},
    "layer_agent_5": {"model_name": "mistral-nemo:12b-instruct-2407-q6_K"},
}

# Really you would probably want to use a larger model here
agent = MOAgent.from_config(
    main_model="llama3.1:8b-instruct-q6_K", layer_agent_config=layer_agent_config
)

while True:
    inp = input("\nAsk a question: ")
    stream = agent.chat(inp, output_format='json')
    for chunk in stream:
        print(chunk)
