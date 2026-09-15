from app.agent.router import RequestAnalyzer


def main() -> None:
    analyzer = RequestAnalyzer()

    test_cases = [
        "Hey MAI, how are you?",
        "What is 4567 multiplied by 892?",
        "What are the latest developments in AI?",
        "What is my preferred language for DSA?",
        "Design a scalable multi-agent architecture for MAI.",
        "Find the best laptop for me.",
    ]

    for request in test_cases:
        print("\nUSER:")
        print(request)

        decision = analyzer.analyze(request)

        print("\nDECISION:")
        print(decision.model_dump_json(indent=2))


if __name__ == "__main__":
    main()