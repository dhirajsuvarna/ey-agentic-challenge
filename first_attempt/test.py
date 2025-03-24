from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.models import UserMessage
import os
from dotenv import load_dotenv
import asyncio

load_dotenv(override=True)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

client = AzureOpenAIChatCompletionClient(
    model="gpt-4o",
    api_version=AZURE_OPENAI_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    temperature=0,
    # model_capabilities={
    #     "vision": False,
    #     "function_calling": True,
    #     "json_output": False,
    # },
)


async def main():

    result = await client.create([UserMessage(content="What is the capital of France?", source="user")])  # type: ignore
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
