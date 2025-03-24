import chainlit as cl
import asyncio
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
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from openai import AzureOpenAI
from autogen_ext.models.openai import (
    AzureOpenAIChatCompletionClient,
)  # Azure OpenAI API
from autogen_agentchat.agents import AssistantAgent  # Agent
from autogen_agentchat.agents import UserProxyAgent  # Agent
from autogen_agentchat.conditions import TextMentionTermination  # Termination condition
from autogen_agentchat.teams import RoundRobinGroupChat  # Agent design pattern
from autogen_agentchat.ui import Console  # To view agent outputs
from tavily import TavilyClient  # For our search tool (Just an example tool)
from dotenv import (
    load_dotenv,
)  # You should have your own .env file with required API keys
import os

# from IPython.display import display, Markdown
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from reportlab.lib.utils import ImageReader
import textwrap
import httpx
from md_to_pdf import convert_md_to_pdf
from pathlib import Path


# # Load environment variables
load_dotenv()

# Retrieve API Keys
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# Initialize Tavily API client
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# # Initialize Chainlit Console
# cl = Console()

client = AzureOpenAIChatCompletionClient(
    model="gpt-4o",
    api_version=AZURE_OPENAI_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    temperature=0,
)


def search_tool(query: str) -> str:
    answer = tavily_client.search(query=query)
    return answer


# Define User Orchestrator
orchestrator = UserProxyAgent(
    name="market_research_orchestrator",
    description="Manages and coordinates market research tasks.",
)


competitor_agent = AssistantAgent(
    name="competitor_research_agent",
    model_client=client,
    system_message=(
        "Analyze top 5 competitors in the market, their market share and strategies. "
        "Feel free to tabulate your results."
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
# technology_agent = AssistantAgent(
#     name="technology_research_agent",
#     model_client=client,
#     system_message="Research emerging technologies affecting this industry.",
# )
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
# swot_agent = AssistantAgent(
#     name="swot_analysis_agent",
#     model_client=client,
#     system_message="Perform a SWOT analysis for the market.",
# )
# report_agent = AssistantAgent(
#     name="report_generator_agent",
#     model_client=client,
#     system_message=" Summarize the market research findings into a structured market research report, be as detailed as possible. Ensure that this report is professional with proper formatting and clearly categorized sections. The report should include an executive summary, market overview, competitor analysis, industry trends, regulatory considerations, technology trends, consumer behavior, SWOT analysis, and recommendations. It should be ready to be saved in PDF format. Say TERMINATE when you are done with summarization",
# )

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


# Define the Round-Robin Agent Group
agent_group = RoundRobinGroupChat(
    participants=[
        industry_agent,
        regulation_agent,
        competitor_agent,
        consumer_agent,
        report_agent,
    ],
    # model_client=client,
    termination_condition=TextMentionTermination("TERMINATE"),
    # max_turns=1,  # This ensures that each agent responds only once.
)


# Store user responses
USER_DETAILS_KEY = "user_details"


@cl.on_chat_start
async def start_chat():
    """Initiates conversation by collecting details from the user before starting research."""
    cl.user_session.set(USER_DETAILS_KEY, {})  # Store user details

    await cl.Message(
        (
            "Hi! Welcome to the **Marketing Research and Strategy Recommender Tool**.\n\n"
            "This tool simplifies the product marketing process by generating a "
            "comprehensive market report tailored to your product. It provides key "
            "insights on competitors, industry trends, consumer behavior, and more, "
            "helping make data driven marketing content. Before we begin, "
            "I’ll need a few details to customize the report for your specific needs. "
            "Let’s get started"
        )
    ).send()

    # Ask the first question
    await cl.Message("Whats the product you are thinking about?").send()


@cl.on_message
async def handle_message(message: cl.Message):
    """Handles user input and collects details before triggering research."""
    user_details = cl.user_session.get(USER_DETAILS_KEY)

    if "product" not in user_details:
        user_details["product"] = message.content
        await cl.Message(
            "Which geographic region should we focus on? (e.g., Global, USA, Europe, Asia)"
        ).send()
        return

    if "region" not in user_details:
        user_details["region"] = message.content
        await cl.Message(
            "Perfect! Now generating your market research report. Please wait..."
        ).send()

        # Run the market research function
        research_results = await execute_market_research(user_details)

        # Display the final report
        url_list = list()
        for result in research_results:
            print(f"dhiraj: {result.source}")
            print(f"type: {type(result)}")
            if result.source != "user":
                if result.source in [
                    "competitor_research_agent",
                    "consumer_research_agent",
                ]:
                    if isinstance(result, (ToolCallExecutionEvent)):
                        for execution_result in result.content:
                            print(f"dhiraj inside: {result.source}")
                            print(f"type of data: {type(execution_result.content)}")
                            tool_results = ast.literal_eval(execution_result.content)
                            for tool_result in tool_results["results"]:
                                print(f"title: {tool_result['title']}")
                                print(f"url: {tool_result['url']}")
                                url_list.append(tool_result["url"])

                    if isinstance(result, TextMessage):
                        if url_list:
                            url_md_string = "\n".join(
                                f"- [{url}]({url})" for url in url_list
                            )
                            result.content = (
                                result.content + "\n\nSources:\n" + url_md_string
                            )
                if isinstance(result, TextMessage):
                    await cl.Message(
                        f"**{result.source}:**\n\n{result.content}",  # elements=elements
                    ).send()

        # Store results in session memory
        cl.user_session.set("research_results", research_results)
        cl.user_session.set("user_details", user_details)
        return

    #  Check if User Wants PDF
    # if "pdf_request" not in user_details:
    #     if message.content.lower() in ["yes", "Yes", "y"]:
    #         user_details["pdf_request"] = True
    #         # pdf_path = generate_pdf_report(
    #         #     user_details, cl.user_session.get("research_results")
    #         # )
    #         pdf_path = "dhiraj"
    #         if os.path.exists(pdf_path):
    #             await cl.Message("PDF generated! Click below to download.").send()
    #             await cl.File(pdf_path, name="Market_Research_Report.pdf").send()
    #             await cl.Message(
    #                 "Would you like to analyze another market? (Yes/No)"
    #             ).send()
    #         else:
    #             await cl.Message("⚠️ Error generating the PDF. Please try again.").send()
    #     else:
    #         await cl.Message(
    #             "No problem! Would you like to analyze another market? (Yes/No)"
    #         ).send()
    #     return
    # # Restart or End Chat
    # if message.content.lower() in ["yes", "Yes", "y"]:
    #     await start_chat()  # Restart the workflow
    # else:
    #     await cl.Message(
    #         "Thank you for using the **Marketing Strategizer Tool**! Have a great day."
    #     ).send()


# async def execute_market_research(user_details):
#     """Runs market research using Round-Robin Agent Chat asynchronously."""
#     industry = user_details["industry"]
#     competitors = user_details["competitors"]
#     market_type = user_details["market_type"]
#     include_swot = user_details["include_swot"]
#     region = user_details["region"]
#     task = f"""
#     Conduct market research on {industry} in {region}.
#     - Competitors: {competitors}
#     - Market Type: {market_type}
#     - Include SWOT Analysis: {include_swot}
#     - Compile insights into a professional report.
#     """

#     task_result = await agent_group.run(task=task)
#     return task_result.messages


async def execute_market_research(user_details):
    """Runs market research using Round-Robin Agent Chat asynchronously."""
    product = user_details["product"]
    region = user_details["region"]
    task = (
        f"I am trying to launch a product {product} in {region}.\n"
        "Help me out with some research."
    )
    task_result = await agent_group.run(task=task)
    return task_result.messages
