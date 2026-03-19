import asyncio
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio


# load the env variables form .env file
load_dotenv()

# update the path with your machine details
server = MCPServerStdio(
    command="/Users/sid/.local/bin/uv",
    args=[
        "--directory",
        "/Volumes/MyDrive/3_udemy/3_udemy_src/2_rec_code/9_mcp/9_2_build_an_mcp_server",
        "run",
        "weather.py",
    ],
)

agent = Agent(
    model="groq:llama-3.3-70b-versatile",
    toolsets=[server],
)

async def main():
    async with agent:
        result = await agent.run("What is the current weather in Chennai?")
        # result = await agent.run(
        #     "What is the current stock price of Tesla?")  # this won't be answered with the weather tool
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
