from app.core.conversation import Conversation
from app.llm.ollama_provider import OllamaProvider
from app.prompts.conversation import MAI_SYSTEM_PROMPT


def main() -> None:
    llm = OllamaProvider()
    conversation = Conversation()

    # First user message
    conversation.add_user_message(
        "My name is Suriya and I'm building a personal AI assistant called MAI."
    )

    response = llm.generate(
        conversation.get_messages(),
        system_prompt=MAI_SYSTEM_PROMPT,
    )

    conversation.add_assistant_message(response)

    print("\nMAI:")
    print(response)

    # Second user message
    conversation.add_user_message(
        "What is my name and what project am I building?"
    )

    response = llm.generate(
        conversation.get_messages(),
        system_prompt=MAI_SYSTEM_PROMPT,
    )

    conversation.add_assistant_message(response)

    print("\nMAI:")
    print(response)


if __name__ == "__main__":
    main()