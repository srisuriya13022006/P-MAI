import os
from app.llm.groq_provider import GroqProvider


def test_real_groq_api():

    provider = GroqProvider(
        model="openai/gpt-oss-20b"
    )

    result = provider.generate(
        [
            {"role": "user", "content": "tell about madurai in one line"}
        ],
        system_prompt="You are a helpful assistant."
    )

    print(result)

    assert result


if __name__ == "__main__":
    test_real_groq_api()