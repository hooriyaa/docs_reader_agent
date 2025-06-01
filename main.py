import chainlit as cl
from dotenv import load_dotenv
import os
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner

# Load .env variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in your .env file.")

# Setup model provider and model
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
)

model = OpenAIChatCompletionsModel(
    openai_client=provider,
    model="gemini-2.0-flash",
)

run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)

# Define your Docs Reader Bot agent
agent1 = Agent(
    name="Docs Reader Bot",
    instructions="""
You are a helpful Docs Reader Bot.

Your job is to:
1. Summarize any document or text in simple English.
2. Translate or explain in Urdu if user asked otherwise answer in english.
3. Explain technical terms or code in an easy way.

Always be clear, kind, and easy to understand.

If a user asks something unrelated (like coding help, math problems, or general questions),
politely respond with:

"I'm here to help with reading, summarizing, and explaining documents. Please share something you'd like help with!" 😊
""",
)


# Chainlit event: On chat start
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="👋 Hi! Paste your document text here and I wll summarize and explain it for you.",
    ).send()


# Chainlit event: On message received
@cl.on_message
async def handle_message(message:cl.Message):
    history = cl.user_session.get("history")
    msg=cl.Message(content="")
    await msg.send()

    history.append({"role": "user", "content":message.content})
    result= Runner.run_streamed(
        agent1,
        input=history,
        run_config=run_config,
    )
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    history.append({"role": "assistant", "content" : result.final_output})
    cl.user_session.set("history", history)

    # command to run the app with Chainlit
# uv run chainlit run main.py -w

