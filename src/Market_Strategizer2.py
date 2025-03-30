import asyncio
import streamlit as st
import ast
from autogen_ext.models.openai import (
    AzureOpenAIChatCompletionClient,
    OpenAIChatCompletionClient,
)
from autogen_agentchat.messages import (
    ToolCallSummaryMessage,
    TextMessage,
    ToolCallExecutionEvent,
)
import json
import autogen_agentchat
import autogen
from dotenv import (
    load_dotenv,
)  
import os
os.environ["AUTOGEN_USE_DOCKER"] = "0"
load_dotenv()
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen import GroupChat
import asyncio


# Retrieve API Keys
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

from tavily import TavilyClient
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

client = AzureOpenAIChatCompletionClient(
    model="gpt-4o",
    api_version=AZURE_OPENAI_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    temperature=0,
)

# )
llm_config={
            "api_type": "azure",
            "api_version":"2025-01-01-preview",
            "model": "gpt-4o",
            "api_key": "52220aca3ae5498fbe5ad757f139e6c4",  
            "base_url": "https://ue2coai4l3aoa02.openai.azure.com/"
            }

def search_tool(query: str) -> str:
    answer = tavily_client.search(query=query)
    return answer

config_list = [{
        "api_type": "azure",
        "api_version":"2025-01-01-preview",
        "model": "gpt-4o",
        "api_key": "52220aca3ae5498fbe5ad757f139e6c4",  
        "base_url": "https://ue2coai4l3aoa02.openai.azure.com/"
        }]

# === AGENT SETUP ===
# I am using the autogen model for the user proxy agent
user_proxy = autogen.UserProxyAgent(
    name = "user_proxy",
    description="Manages and coordinates market research tasks.",
    human_input_mode='NEVER',)

# Defining the specialized agents for all other tasks
competitor_agent = AssistantAgent(
    name="competitor_research_agent",
    model_client=client,
    system_message=(
        "Analyze top 5 of the competitors in the market for this product, their market share and strategies. "
        "Please present your results in a table."
    ),
    tools=[search_tool],
    reflect_on_tool_use=True,
)

industry_agent = AssistantAgent(
    name="industry_trends_agent",
    model_client=client,
    system_message=(
        "For the given product identify the industry it belongs to "
        "and provide an industry trend analysis containing following sections-\n"
        "- Industry Overview\n"
        "- Industry Disruptions\n"
        "- Industry Innovations"
    ),
)
regulation_agent = AssistantAgent(
    name="regulation_research_agent",
    model_client=client,
    system_message=(
        "You are a Regulatory Complaiance Manager. You will be given a "
        "product and the location where it will be launched. Your task is to figure out "
        "various compliance that needs to be followed while launching the product "
        "geography."
    ),
)

consumer_agent = AssistantAgent(
    name="consumer_research_agent",
    model_client=client,
    system_message=(
        "Generte Consumer Insights and Target Audience Analysis based on the product that"
        "is being launched and also the geography where it is intended to launch.\n"
        "The analysis should include -\n"
        "- Consumer Demographics\n"
        "- Buying patterns\n"
        "- Customer pain points and needs\n"
        "Use the tools you have to perform the research on these topics."
    ),
    tools=[search_tool],
    reflect_on_tool_use=True,
)

report_agent = AssistantAgent(
    name="strategy_reommender_agent",
    model_client=client,
    system_message=(
        "Analyze the market research findings and provide Marketing Strategy Recommendation "
        "based on the topic. "
        "The recomendation should include -\n"
        "- Some Key Features that the product should include, depnding on research that has been done"
        "- List of 3 USP that the product should be having\n"
        "- List of 3 Brand names that would appeal to the target audience\n"
        "- Recommend the Target Audience for this product\n"
        "- Price Point\n"
        "- Tone for the Ad\n"
        "Tabulate the above recommendation in following format - \n"
        "Brand Name, Key Features, Target Audience, Price Point, Tone of Ad"
        "Say TERMINATE when you are done."
    ),
)


# Create the round-robin group chat
agent_group = RoundRobinGroupChat(
    participants=[
        industry_agent,
        regulation_agent,
        competitor_agent,
        consumer_agent,
        report_agent,
    ],
    termination_condition=TextMentionTermination("TERMINATE")
    )
# === STREAMLIT SESSION STATE INITIALIZATION ===
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_proxy" not in st.session_state:
    st.session_state.user_proxy = autogen.UserProxyAgent(name = "user_proxy")
if "groupchat_ran" not in st.session_state:
    st.session_state.groupchat_ran = False

if "orchestrator" not in st.session_state:
   st.session_state.orchestrator = autogen.AssistantAgent(
            name ="market_research_orchestrator", 
            description="Guides the user through market research questions.",
            llm_config= llm_config, 
            system_message = (
            "You are a helpful market research assistant. Start by greeting the user with:\n\n"
            "'Hi! Welcome to the Marketing Research and Strategy Recommender Tool. This tool simplifies the product marketing process by generating a comprehensive market report tailored to your product. "
            "It provides key insights on competitors, industry trends, consumer behavior, and more, helping make data-driven marketing content. "
            "Before we begin, I’ll need a few details to customize the report for your specific needs. Let’s get started.'\n\n"
            "Then ask:\n"
            "1. What's the product you are thinking about?\n"
            "Once the user answers, ask a clarifying question about the use-case.\n"
            "Then ask:\n"
            "2. Which geographic region should we focus on? (e.g., Global, USA, Europe, Asia)\n"
            "After that, acknowledge and say you'll begin analysis."
        )
    )
if "product" not in st.session_state:
    st.session_state.product = None

if "region" not in st.session_state:
    st.session_state.region = None

if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0

user_proxy = st.session_state.user_proxy
orchestrator = st.session_state.orchestrator

# === UI header ===
st.title("Marketing Research and Strategy Assistant")

# === Display past messages ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === ASYNC STREAMING AGENT EXECUTION ===
def execute_market_research_stream_sync(user_details, progress_bar, progress_state, chat_container):
    product = user_details["product"]
    region = user_details["region"]
    task = (
        f"I am trying to launch a product '{product}' in '{region}'.\n"
        "Help me out with some market research."
    )

    agent_group.reset()  # Clear old state
    agent_group.add_user_message(task)

    while not agent_group.is_terminated():
        step_messages = agent_group.step()

        if step_messages:
            for msg in step_messages:
                name = msg.get("name", "agent")
                content = msg.get("content", "")

                with chat_container.chat_message(name):
                    st.markdown(content)  # ✅ Stream output live

                progress_state["completed"] += 1
                progress = min(1.0, progress_state["completed"] / progress_state["total"])
                progress_bar.progress(progress)

        st.experimental_rerun()  # ✅ Optional for live UI refresh


# === CHAT INPUT HANDLING ===
if prompt := st.chat_input("Enter your message"):
    st.session_state.input_counter += 1
    user_input = prompt.lower()

    # Save the second user input as product
    if st.session_state.input_counter == 2 and st.session_state.product is None:
        st.session_state.product = prompt
        st.toast(f"✅ Product saved: {st.session_state.product}")

    # Save as region if user mentions a known geography keyword
    if st.session_state.region is None and any(loc in user_input for loc in ["usa", "europe", "asia", "global"]):
        st.session_state.region = prompt
        st.toast(f"🌍 Region saved: {st.session_state.region}")

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # === Construct full chat history ===
    chat_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    # === Generate response from the orchestrator ===
    response = orchestrator.generate_reply(
        messages=chat_history,
        sender=user_proxy,
    )
    
    if isinstance(response, dict) and "content" in response:
        reply = response["content"]
    elif isinstance(response, str):
        reply = response
    else:
        reply = "*[No response generated — check your config/model.]*"
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)

    if st.button("🚀 Run Market Research") and not st.session_state.groupchat_ran:
        st.session_state.groupchat_ran = True

        user_details = {
            "product": st.session_state.product,
            "region": st.session_state.region,
        }

        st.subheader("📡 Live Research")
        chat_container = st.container()
        progress_bar = st.progress(0)
        progress_state = {"completed": 0, "total": len(agent_group.participants)}

        with st.spinner("Agents are thinking..."):
            execute_market_research_stream_sync(
                user_details=user_details,
                progress_bar=progress_bar,
                progress_state=progress_state,
                chat_container=chat_container
            )

        st.success("✅ Research complete!")
