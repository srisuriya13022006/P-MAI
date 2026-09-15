from app.agent.orchestrator import MAIOrchestrator


def main() -> None:
    orchestrator = MAIOrchestrator()

    response = orchestrator.handle(
        user_message="What is 45 * 87?",
        conversation_messages=[
            {
                "role": "user",
                "content": "What is 45 * 87?",
            }
        ],
    )

    print("\nMAI:")
    print(response)


if __name__ == "__main__":
    main()