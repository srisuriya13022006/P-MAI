from app.memory.summarizer import ConversationSummarizer


class FakeLLM:
    def generate(self, messages, system_prompt=None):
        assert messages[0]["role"] == "user"
        assert "Summarize this conversation" in system_prompt
        return "User is building MAI."


def test_summarizer_delegates_with_factual_prompt():
    summarizer = ConversationSummarizer(FakeLLM())

    result = summarizer.summarize(
        [
            {"role": "user", "content": "I am building MAI."},
            {"role": "assistant", "content": "Great."},
        ]
    )

    assert result == "User is building MAI."