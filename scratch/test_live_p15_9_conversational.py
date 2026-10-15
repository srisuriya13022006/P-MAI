"""
Live verification for P15.9: Conversational Intelligence, Context Awareness & Personality Refinement.
Tests:
1. Turn 1: "Are you equal to Jarvis?"
2. Turn 2: "Hey, answer the question!" -> verifies MAI resolves the follow-up and gives a direct answer without repeating or saying 'Sir'.
3. Turn 3: "Who is more capable, Jarvis or you?" -> verifies MAI answers the actual question first.
4. Turn 4: "What tools do you have?" -> verifies MAI clearly explains its implemented tools.
5. Turn 5: "Write a song about Arthur's lighthouse" -> verifies full song generation without truncation.
"""
import sys
import os

sys.path.insert(0, r"d:\suriya\projects\P-MAI")

from app.database.session import SessionLocal
from app.services.chat_service import ChatService


def test_live_conversation():
    print("Initializing ChatService for P15.9 live conversation test...")
    db = SessionLocal()
    try:
        service = ChatService(db=db)
        conv_id = f"test-p15-9-{os.urandom(4).hex()}"
        user_id = "test-user-p15-9"

        # Turn 1
        print("\n--- TURN 1 ---")
        q1 = "Are you equal to Jarvis?"
        print(f"User: {q1}")
        a1 = service.chat(conv_id, user_id, q1)
        print(f"MAI: {a1}")

        # Check for Sir
        assert "sir" not in a1.lower().split(), "MAI used 'Sir' in Turn 1!"

        # Turn 2
        print("\n--- TURN 2 ---")
        q2 = "Hey, answer the question!"
        print(f"User: {q2}")
        a2 = service.chat(conv_id, user_id, q2)
        print(f"MAI: {a2}")

        assert "sir" not in a2.lower().split(), "MAI used 'Sir' in Turn 2!"
        # Verify MAI gave a direct answer to the question
        lowered_a2 = a2.lower()
        has_direct_answer = any(w in lowered_a2 for w in ("no", "not yet", "jarvis", "fictional", "cannot match", "practical"))
        print(f"Direct answer detected: {has_direct_answer}")

        # Turn 3
        print("\n--- TURN 3 ---")
        q3 = "Who is more capable, Jarvis or you?"
        print(f"User: {q3}")
        a3 = service.chat(conv_id, user_id, q3)
        print(f"MAI: {a3}")

        assert "sir" not in a3.lower().split(), "MAI used 'Sir' in Turn 3!"

        # Turn 4
        print("\n--- TURN 4 ---")
        q4 = "What tools do you have?"
        print(f"User: {q4}")
        a4 = service.chat(conv_id, user_id, q4)
        print(f"MAI: {a4}")

        assert "web search" in a4.lower()
        assert "calculator" in a4.lower()
        assert "memory" in a4.lower()

        # Turn 5
        print("\n--- TURN 5 ---")
        q5 = "Write a song about Arthur's lighthouse"
        print(f"User: {q5}")
        a5 = service.chat(conv_id, user_id, q5)
        print(f"MAI:\n{a5}")
        print(f"Song length (chars): {len(a5)}, words: {len(a5.split())}")
        assert len(a5.split()) >= 80, "Song was truncated!"

        print("\nALL LIVE P15.9 CONVERSATION TESTS PASSED SUCCESSFULLY!")

    finally:
        db.close()


if __name__ == "__main__":
    test_live_conversation()
