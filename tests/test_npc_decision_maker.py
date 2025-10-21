from npcs.npc_decision_maker_module import NPCDecisionMaker
import unittest
from unittest.mock import Mock, patch


class TestNPCDecisionMaker(unittest.TestCase):
    def setUp(self):
        self.npc_dm = NPCDecisionMaker()
        self.test_npc = {
            "name": "Test NPC",
            "personality": "Test personality",
            "situation": "Test situation",
            "player_action": "Test action",
            "context": "Test context"
        }

    def test_create_npc_prompt(self):
        prompt = self.npc_dm.create_npc_prompt(
            self.test_npc["name"],
            self.test_npc["personality"],
            self.test_npc["situation"],
            self.test_npc["player_action"],
            self.test_npc["context"]
        )
        self.assertIsInstance(prompt, str)
        self.assertIn(self.test_npc["name"], prompt)
        self.assertIn(self.test_npc["context"], prompt)

    @patch('requests.post')
    def test_get_npc_response(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": '{"dialogue": "Test dialogue", "actions": "Test actions", "emotion": "Test emotion", "decision": "Test decision"}'
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        response = self.npc_dm.get_npc_response(
            self.test_npc["name"],
            self.test_npc["personality"],
            self.test_npc["situation"],
            self.test_npc["player_action"]
        )

        self.assertIsInstance(response, dict)
        self.assertIn("dialogue", response)

    @patch('requests.post')
    def test_get_npc_response_streaming(self, mock_post):
        mock_response = Mock()
        mock_response.iter_lines.return_value = [
            b'{"response": "chunk1"}',
            b'{"response": "chunk2"}'
        ]
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        chunks = list(self.npc_dm.get_npc_response_streaming(
            self.test_npc["name"],
            self.test_npc["personality"],
            self.test_npc["situation"],
            self.test_npc["player_action"]
        ))

        self.assertTrue(len(chunks) > 0)
        self.assertIsInstance(chunks[0], str)


if __name__ == "__main__":
    unittest.main()