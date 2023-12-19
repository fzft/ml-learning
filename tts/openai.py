from pathlib import Path

import openai
from packaging import version

required_version = version.parse("1.1.1")
current_version = version.parse(openai.__version__)

if current_version < required_version:
    raise ValueError(f"Error: OpenAI version {openai.__version__}"
                     " is less than the required version 1.1.1")
else:
    print("OpenAI version is compatible.")

from openai import OpenAI

if __name__ == '__main__':
    client = OpenAI(api_key="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")

    speech_file_path = Path(__file__).parent / "linzhiling.mp3"
    response = client.audio.speech.create(
      model="tts-1",
      voice="alloy",
      input="你好帅哥，周末有空么?"
    )

    response.stream_to_file("speech_file_path")