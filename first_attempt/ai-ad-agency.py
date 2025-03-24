#Getting the imports
import os, json, string, uuid
from typing import List
import base64,cv2
import openai
import asyncio
from openai import AzureOpenAI
from openai import OpenAI

from autogen_core import (
    DefaultTopicId,
    FunctionCall,
    Image,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from IPython.display import display
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
from pyht import Client
from pyht.client import TTSOptions
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from dotenv import load_dotenv
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
import json
from typing import List

#Loading Env Variables
load_dotenv(override=True)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT= os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION")
OPENAI_API_KEY_PERSONAL = os.getenv("OPENAI_API_KEY_PERSONAL")

#Initializing the model client
personal_model_client = OpenAIChatCompletionClient(
            model="gpt-4o-2024-08-06",
            api_key=OPENAI_API_KEY_PERSONAL,
        )

ey_model_client = model_client=AzureOpenAIChatCompletionClient(
        model="gpt-4o",
        api_version=AZURE_OPENAI_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        temperature=0,
    )

#BaseGroupAgent

class GroupChatMessage(BaseModel):
    body: UserMessage

class RequestToSpeak(BaseModel):
    pass

class BaseGroupChatAgent(RoutedAgent):
    """A group chat participant using an LLM."""

    def __init__(
        self,
        description: str,
        group_chat_topic_type: str,
        model_client: ChatCompletionClient,
        system_message: str,
    ) -> None:
        super().__init__(description=description)
        self._group_chat_topic_type = group_chat_topic_type
        self._model_client = model_client
        self._system_message = SystemMessage(content=system_message)
        self._chat_history: List[LLMMessage] = []

    @message_handler
    async def handle_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        self._chat_history.extend(
            [
                UserMessage(content=f"Transferred to {message.body.source}", source="system"),
                message.body,
            ]
        )

    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        Console().print(Markdown(f"### {self.id.type}: "))
        self._chat_history.append(
            UserMessage(content=f"Transferred to {self.id.type}, adopt the persona immediately.", source="system")
        )
        completion = await self._model_client.create([self._system_message] + self._chat_history)
        assert isinstance(completion.content, str)
        self._chat_history.append(AssistantMessage(content=completion.content, source=self.id.type))
        Console().print(Markdown(completion.content))
        await self.publish_message(
            GroupChatMessage(body=UserMessage(content=completion.content, source=self.id.type)),
            topic_id=DefaultTopicId(type=self._group_chat_topic_type),
        )


##### Write Agent
class WriterAgent(BaseGroupChatAgent):
    def __init__(self, description: str, group_chat_topic_type: str, model_client: ChatCompletionClient) -> None:
        super().__init__(
            description=description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
            system_message="You are a Writer. You produce good work. For audio you give one script and not multiple for each image",
        )


##### Editor Agent
class EditorAgent(BaseGroupChatAgent):
    def __init__(self, description: str, group_chat_topic_type: str, model_client: ChatCompletionClient) -> None:
        super().__init__(
            description=description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
            system_message="You are an Editor. Plan and guide the task given by the user. Provide critical feedbacks to the draft and illustration produced by Writer and Illustrator. "
            "Approve if the task is completed and the draft and illustration meets user's requirements.",
        )


##### Illustrator Agent

#### Image Model Code
 
class IllustratorAgent(BaseGroupChatAgent):
    def __init__(
        self,
        description: str,
        group_chat_topic_type: str,
        model_client: ChatCompletionClient,
    ) -> None:
        super().__init__(
            description=description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
            system_message="You are an Illustrator. You use the generate_image tool to create images given user's requirement. "
            "Make sure the images have consistent characters and style.",
        )
        
       
        # Initialize the image generation tool
        self._image_gen_tool = FunctionTool(
            self._image_gen, name="generate_image", description="Call this to generate an image. "
        )
 
        # If openai:
        #self._image_client = openai.AsyncClient

        #If huggingface:
        #  Load the model for image generation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            variant="fp16"
        ).to(self.device)

 
    async def _image_gen(
        self, character_appearance: str, style_attributes: str, worn_and_carried: str, scenario: str
    ) -> str:
        prompt = f"Digital painting of a {character_appearance} character with {style_attributes}. Wearing {worn_and_carried}, {scenario}."
        #if openai:
        # response = await self._image_client.images.generate(
        #     prompt=prompt, model="dall-e-3", response_format="b64_json", size="1024x1024"
        # )
        # return response.data[0].b64_json
    
        # if huggingface
        image = self.pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
        return image 
 
    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:  # type: ignore
        Console().print(Markdown(f"### {self.id.type}: "))
        self._chat_history.append(
            UserMessage(content=f"Transferred to {self.id.type}, adopt the persona immediately. Avoid any content that may be considered inappropriate or offensive, ensuring the image aligns with content policies.", source="system")
        )
       
        # Ensure that the image generation tool is used.
        completion = await self._model_client.create(
            [self._system_message] + self._chat_history,
            tools=[self._image_gen_tool],
            extra_create_args={"tool_choice": "required"},
            cancellation_token=ctx.cancellation_token,
        )
       
        assert isinstance(completion.content, list) and all(
            isinstance(item, FunctionCall) for item in completion.content
        )
       
        #if huggingface
        images: List[str] = []

        # if openai
        # images: List[str | Image] = []
        for i, tool_call in enumerate(completion.content):
            arguments = json.loads(tool_call.arguments)
            Console().print(arguments)
            result = await self._image_gen_tool.run_json(arguments, ctx.cancellation_token)

            #if huggingface
            image = result  
            image_path = f"image_{i+1}.png"
            image.save(image_path)
            images.append(image_path)
            image_resized = image.resize((256, 256))
            display(image_resized)

            #if openai:
            #image = Image.from_base64(self._image_gen_tool.return_value_as_string(result))
            #image = Image.from_pil(image.image.resize((256, 256)))
            #display(image.image)
            #image.image.save(f"image_{i+1}.png")

        await self.publish_message(
            GroupChatMessage(body=UserMessage(content=images, source=self.id.type)),
            DefaultTopicId(type=self._group_chat_topic_type),
        )
       

import base64
from openai import OpenAI
from pyht import Client
from dotenv import load_dotenv
from pyht.client import TTSOptions
import os
load_dotenv(override=True)

##### TTS Agent
class TextToSpeechAgent(BaseGroupChatAgent):
    def __init__(
        self,
        description: str,
        group_chat_topic_type: str,
        model_client: ChatCompletionClient,
    ) -> None:
        super().__init__(
            description=description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
            system_message="You are a Text-to-Speech Agent. You convert text into spoken audio using the generate_audio tool.",
        )
        self._audio_gen_tool = FunctionTool(
            self._audio_gen, name="generate_audio", description="Call this to generate audio from text."
        )


        self.audio_openai_client = OpenAI(api_key=OPENAI_API_KEY_PERSONAL)


    async def _audio_gen(self, text: str) -> str:
        completion = self.audio_openai_client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {
                "role": "user",
                "content": f"Convert the following text to speech: '{text}'"
            }
        ]
    )
        wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)
        with open("output.wav", "wb") as f:
            f.write(wav_bytes)

        return completion.choices[0]

    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        Console().print(Markdown(f"### {self.id.type}: Received message from TTS topic"))
        self._chat_history.append(
            UserMessage(content=f"Transferred to {self.id.type}, adopt the persona immediately.", source="system")
        )
        # Ensure that the audio generation tool is used.
        completion = await self._model_client.create(
            [self._system_message] + self._chat_history,
            tools=[self._audio_gen_tool],
            extra_create_args={"tool_choice": "required"},
            cancellation_token=ctx.cancellation_token,
        )
        Console().print(completion.content)
        assert isinstance(completion.content, list) and all(
            isinstance(item, FunctionCall) for item in completion.content
        )
        audio_urls: List[str] = []
        for tool_call in completion.content:
            arguments = json.loads(tool_call.arguments)
            Console().print(arguments)
            result = await self._audio_gen_tool.run_json(arguments, ctx.cancellation_token)

            audio_urls.append(result)
        await self.publish_message(
            GroupChatMessage(body=UserMessage(content=f"Audio for the ad published here", source=self.id.type)),
            DefaultTopicId(type=self._group_chat_topic_type),
        )
        

#Animator Agent
class AnimatorAgent(BaseGroupChatAgent):
    def __init__(
        self,
        description: str,
        group_chat_topic_type: str,
        model_client: ChatCompletionClient,
    ) -> None:
        super().__init__(
            description=description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
            system_message="You are an Animation Agent. You create videos from images and add audio.",
        )
        self._animation_tool = FunctionTool(
            self._create_video_with_moviepy_workaround, name="create_video_with_moviepy_workaround", description="Call this to create a video from images and add audio."
        )

    
    async def _create_video_with_moviepy_workaround(self,image_paths: List[str]) -> str:
        """
        Creates a video from images and adds audio using moviepy.
        Fixed to properly handle duration calculations.
        Only run it after the TextToSpeech Agent has produced the audio and Editor has APPROVED the work.
        """
        
        # Hardcoded paths for testing
        image_paths = [
            "image_1.png",
            "image_2.png",
            "image_3.png"
        ]
        audio_path = "output.wav"
        output_path = "output_video.mp4"

        # Convert to absolute paths
        #image_paths = [os.path.abspath(img_path) for img_path in image_paths]
        # audio_path = os.path.abspath(audio_path)
        # output_path = os.path.abspath(output_path)

        # Get audio duration   
        audio_clip = AudioFileClip(audio_path)
        audio_duration = audio_clip.duration

        print(f"Audio duration: {audio_duration} seconds")
        
        # Calculate how long each image should be shown
        num_images = len(image_paths)
        image_duration = audio_duration / num_images
        
        # Load images and resize to same dimensions
        images = [cv2.imread(img, cv2.IMREAD_UNCHANGED) for img in image_paths]
        height, width, _ = images[0].shape
        images = [cv2.resize(img, (width, height)) for img in images]

        # Convert images to RGB (moviepy expects RGB format)
        images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGRA2RGB) if img.shape[2] == 4 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        fps=30
        # **Ensure fps is set properly**
        if fps is None or not isinstance(fps, (int, float)):  
            fps = 30  # Default to 30 FPS if fps is None or invalid

        # Create video clip with an explicit fps
        clips = []
        for img_path in image_paths:
            # Create an ImageClip with explicit duration
            clip = ImageClip(img_path).set_duration(image_duration)
            clips.append(clip)
        
        # Concatenate all image clips
        clip = concatenate_videoclips(clips)
        
        # Add audio
        audio_clip = AudioFileClip(audio_path)
        clip = clip.set_audio(audio_clip)

        # Write final video file
        clip.write_videofile(output_path, codec="libx264", fps=30, audio_codec="mp3")

        return output_path

    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:  # type: ignore
        Console().print(Markdown(f"### {self.id.type}: Received request to create video with audio"))
        self._chat_history.append(
            UserMessage(content=f"Transferred to {self.id.type}, adopt the persona immediately.", source="system")
        )
        # Ensure that the animation tool is used.
        completion = await self._model_client.create(
            messages=[self._system_message] + self._chat_history,
            tools=[self._animation_tool],
            extra_create_args={"tool_choice": "required"},
            cancellation_token=ctx.cancellation_token,
        )
        assert isinstance(completion.content, list) and all(
            isinstance(item, FunctionCall) for item in completion.content
        )
        video_paths: List[str] = []
        for tool_call in completion.content:
            arguments = json.loads(tool_call.arguments)
            Console().print(arguments)
            result = await self._animation_tool.run_json(arguments, ctx.cancellation_token)
            video_paths.append(result)
        await self.publish_message(
            GroupChatMessage(body=UserMessage(content=video_paths, source=self.id.type)),
            DefaultTopicId(type=self._group_chat_topic_type),
        )


#User Agent
class UserAgent(RoutedAgent):
    def __init__(self, description: str, group_chat_topic_type: str) -> None:
        super().__init__(description=description)
        self._group_chat_topic_type = group_chat_topic_type

    @message_handler
    async def handle_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        # When integrating with a frontend, this is where group chat message would be sent to the frontend.
        pass

    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        user_input = input("Enter product info, to start the task: ")
        Console().print(Markdown(f"### User: \n{user_input}"))
        await self.publish_message(
            GroupChatMessage(body=UserMessage(content=user_input, source=self.id.type)),
            DefaultTopicId(type=self._group_chat_topic_type),
        )

#Approver Agent
class ApproverAgent(RoutedAgent):
    def __init__(self, description: str, group_chat_topic_type: str) -> None:
        super().__init__(description=description)
        self._group_chat_topic_type = group_chat_topic_type

    @message_handler
    async def handle_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        # When integrating with a frontend, this is where group chat message would be sent to the frontend.
        pass

    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        user_input = input("Enter your message, type 'Done' to conclude the task: ")
        Console().print(Markdown(f"### User: \n{user_input}"))
        await self.publish_message(
            GroupChatMessage(body=UserMessage(content=user_input, source=self.id.type)),
            DefaultTopicId(type=self._group_chat_topic_type),
        )

#GroupChatManager
class GroupChatManager(RoutedAgent):
    def __init__(
        self,
        participant_topic_types: List[str],
        model_client: ChatCompletionClient,
        participant_descriptions: List[str],
    ) -> None:
        super().__init__("Group chat manager")
        self._participant_topic_types = participant_topic_types
        self._model_client = model_client
        self._chat_history: List[UserMessage] = []
        self._participant_descriptions = participant_descriptions
        self._previous_participant_topic_type: str | None = None

    @message_handler
    async def handle_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        assert isinstance(message.body, UserMessage)
        self._chat_history.append(message.body)
        # If the message is an approval message from the user, stop the chat.
        if message.body.source == "Approver":
            assert isinstance(message.body.content, str)
            if message.body.content.lower().strip(string.punctuation).endswith("done"):
                Console().print("Process marked as complete")  # **Marked Change**
                return
        # Format message history.
        messages: List[str] = []
        for msg in self._chat_history:
            if isinstance(msg.content, str):
                messages.append(f"{msg.source}: {msg.content}")
            elif isinstance(msg.content, list):
                line: List[str] = []
                for item in msg.content:
                    if isinstance(item, str):
                        line.append(item)
                    else:
                        line.append("[Image]")
                messages.append(f"{msg.source}: {', '.join(line)}")
        history = "\n".join(messages)
        # Format roles.
        roles = "\n".join(
            [
                f"{topic_type}: {description}".strip()
                for topic_type, description in zip(
                    self._participant_topic_types, self._participant_descriptions, strict=True
                )
                if topic_type != self._previous_participant_topic_type
            ]
        )
        selector_prompt = """You are in a role play game. The following roles are available:
{roles}.
Read the following conversation. Then select the next role from {participants} to play. Only return the role.

{history}

Read the above conversation. Then select the next role from {participants} to play. Only return the role.
"""
        system_message = SystemMessage(
            content=selector_prompt.format(
                roles=roles,
                history=history,
                participants=str(
                    [
                        topic_type
                        for topic_type in self._participant_topic_types
                        if topic_type != self._previous_participant_topic_type
                    ]
                ),
            )
        )
        completion = await self._model_client.create([system_message], cancellation_token=ctx.cancellation_token)
        assert isinstance(completion.content, str)
        selected_topic_type: str
        for topic_type in self._participant_topic_types:
            if topic_type.lower() in completion.content.lower():
                selected_topic_type = topic_type
                self._previous_participant_topic_type = selected_topic_type
                await self.publish_message(RequestToSpeak(), DefaultTopicId(type=selected_topic_type))
                return
        raise ValueError(f"Invalid role selected: {completion.content}")
    

async def call_agents():
    runtime = SingleThreadedAgentRuntime()

    user_topic_type = "User"
    writer_topic_type = "Writer"
    editor_topic_type = "Editor"
    illustrator_topic_type = "Illustrator"
    tts_topic_type = "TextToSpeech"
    animator_topic_type = "Animator"
    approver_topic_type = "Approver"


    group_chat_topic_type = "group_chat"


    user_description = "User for provide product information."
    writer_description = "Writer for creating any ad content."
    editor_description = "Editor for planning and reviewing the  ad content."
    illustrator_description = "An illustrator for creating ad images."
    tts_description = "Speaker for converting text to spoken audio."
    animator_description = "An animator for stitching images together."
    approver_description = "User for providing final approval."

    user_agent_type = await UserAgent.register(
        runtime,
        user_topic_type,
        lambda: UserAgent(description=user_description, group_chat_topic_type=group_chat_topic_type),
    )
    await runtime.add_subscription(TypeSubscription(topic_type=user_topic_type, agent_type=user_agent_type.type))
    await runtime.add_subscription(TypeSubscription(topic_type=group_chat_topic_type, agent_type=user_agent_type.type))

    editor_agent_type = await EditorAgent.register(
        runtime,
        editor_topic_type,
        lambda: EditorAgent(
            description=editor_description,
            group_chat_topic_type=group_chat_topic_type,
            model_client = ey_model_client ),
    )
    await runtime.add_subscription(TypeSubscription(topic_type=editor_topic_type, agent_type=editor_agent_type.type))
    await runtime.add_subscription(TypeSubscription(topic_type=group_chat_topic_type, agent_type=editor_agent_type.type))

    writer_agent_type = await WriterAgent.register(
        runtime,
        writer_topic_type,  # Using topic type as the agent type.
        lambda: WriterAgent(
            description=writer_description,
            group_chat_topic_type=group_chat_topic_type,
            model_client = ey_model_client ),
    )
    await runtime.add_subscription(TypeSubscription(topic_type=writer_topic_type, agent_type=writer_agent_type.type))
    await runtime.add_subscription(TypeSubscription(topic_type=group_chat_topic_type, agent_type=writer_agent_type.type))

    illustrator_agent_type = await IllustratorAgent.register(
        runtime,
        illustrator_topic_type,
        lambda: IllustratorAgent(
            description=illustrator_description,
            group_chat_topic_type=group_chat_topic_type,
            model_client = ey_model_client ),
    )
    await runtime.add_subscription(TypeSubscription(topic_type=illustrator_topic_type, agent_type=illustrator_agent_type.type))
    await runtime.add_subscription(TypeSubscription(topic_type=group_chat_topic_type, agent_type=illustrator_agent_type.type))


    # Register TextToSpeechAgent
    tts_agent_type = await TextToSpeechAgent.register(
        runtime,
        tts_topic_type,
        lambda: TextToSpeechAgent(
            description=tts_description,
            group_chat_topic_type=group_chat_topic_type,
            model_client = personal_model_client
            ),
    )
    await runtime.add_subscription(TypeSubscription(topic_type=tts_topic_type, agent_type=tts_agent_type.type))
    await runtime.add_subscription(TypeSubscription(topic_type=group_chat_topic_type, agent_type=tts_agent_type.type))

    approver_agent_type = await ApproverAgent.register(
        runtime,
        approver_topic_type,
        lambda: ApproverAgent(description=approver_description, group_chat_topic_type=group_chat_topic_type),
    )
    await runtime.add_subscription(TypeSubscription(topic_type=approver_topic_type, agent_type=approver_agent_type.type))
    await runtime.add_subscription(TypeSubscription(topic_type=group_chat_topic_type, agent_type=approver_agent_type.type))

    # Register AnimationAgent
    animator_agent_type = await AnimatorAgent.register(
        runtime,
        animator_topic_type,
        lambda: AnimatorAgent(
            description="Animation Agent for creating animations from images.",
            group_chat_topic_type=group_chat_topic_type,
            model_client = ey_model_client 
            ),
    )
    await runtime.add_subscription(TypeSubscription(topic_type=animator_topic_type, agent_type=animator_agent_type.type))
    await runtime.add_subscription(TypeSubscription(topic_type=group_chat_topic_type, agent_type=animator_agent_type.type))


    group_chat_manager_type = await GroupChatManager.register(
        runtime,
        "group_chat_manager",
        lambda: GroupChatManager(
            participant_topic_types=[user_topic_type, writer_topic_type, illustrator_topic_type, editor_topic_type, tts_topic_type,approver_topic_type,animator_topic_type],
            model_client= ey_model_client,
            participant_descriptions=[user_description,writer_description, illustrator_description, editor_description, tts_description,approver_description,animator_description],
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type=group_chat_topic_type, agent_type=group_chat_manager_type.type)
    )

    runtime.start()
    session_id = str(uuid.uuid4())
    await runtime.publish_message(
        GroupChatMessage(
            body=UserMessage(
                #content="First collect product info by asking particular questions regarding the product name, target audio, description etc. then write a 30 second video script for a product ad with up to 3 photo-realistic, generate audio for the illustrations, and stitched illustrations and audio.",
                #content="Please create a video ad of 10 secs, first get product info from the users, then write script, then generate images, then create audio, after than review the script, and finally generate the video after that after the user if user is done",
                #content="Get Product info from user then write a 5 second video script for a product ad with up to 1 photo-realistic illustration, generate the audio for it and stitched the images into a video with audio.",
                content = """Please create a 10-second video ad. Follow these steps:
                            1. Gather Product Information: Ask the user for details about the product.
                            2. Write the Script: Develop a concise script based on the product information.
                            3. Generate Images: Create 3 images that aligns with the script, donot use illustrator agent once the script is approved by the editor.
                            4. Create Audio: Produce or source audio that complements the script and images.
                            5. Review the Script: Reivew the script, be lenient .
                            6. Generate the Video: Onces the Editor Agent approves create the animation, only generate video once you have generated the audio.
                            7. User Confirmation: Ask the user if they are satisfied with the final video.""",
                source="User",
            )
        ),
        TopicId(type=group_chat_topic_type, source=session_id),
    )
    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(call_agents())