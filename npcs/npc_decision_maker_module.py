import requests
import json
from typing import Dict, List, Optional


class NPCDecisionMaker:
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama2"):
        self.ollama_url = ollama_url
        self.model = model
        self.api_endpoint = f"{ollama_url}/api/generate"
    
    def create_npc_prompt(
        self,
        npc_name: str,
        npc_personality: str,
        situation: str,
        player_action: str,
        context: Optional[str] = None
    ) -> str:
        prompt = f"""You are a Dungeon Master assistant for a tabletop RPG game.

NPC Information:
- Name: {npc_name}
- Personality: {npc_personality}

Current Situation: {situation}

Player Action: {player_action}
"""
        
        if context:
            prompt += f"\nAdditional Context: {context}"
        
        prompt += """

Generate a response for this NPC. Include:
1. The NPC's spoken dialogue (in quotes)
2. The NPC's actions or body language (in italics or brackets)
3. Their emotional state
4. Any decisions they make

Keep the response concise and in-character. Format your response as JSON with these fields:
- dialogue: what the NPC says
- actions: what the NPC does
- emotion: how the NPC feels
- decision: what the NPC decides to do next
"""
        return prompt
    
    def get_npc_response(
        self,
        npc_name: str,
        npc_personality: str,
        situation: str,
        player_action: str,
        context: Optional[str] = None,
        temperature: float = 0.7
    ) -> Dict:
        """
        Get an NPC response from Ollama.
        
        Args:
            npc_name: Name of the NPC
            npc_personality: Description of NPC's personality
            situation: Current situation
            player_action: What the player did
            context: Optional additional context
            temperature: Creativity level (0.0-1.0, higher = more creative)
        
        Returns:
            Dictionary containing the NPC's response
        """
        prompt = self.create_npc_prompt(
            npc_name, npc_personality, situation, player_action, context
        )
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "format": "json",
            "stream": False,
            "response-type": "Only respond in valid JSON format.",
        }
        
        try:
            response = requests.post(self.api_endpoint, json=payload)
            response.raise_for_status()
            
            result = response.json()
            llm_response = result.get("response", "")
            
            try:
                parsed_response = json.loads(llm_response)
                return parsed_response
            except json.JSONDecodeError:
                return {
                    "raw_response": llm_response,
                    "dialogue": llm_response,
                    "note": "Response not in JSON format"
                }
        
        except requests.exceptions.RequestException as e:
            print(f"Error: {str(e)}")
            return {
                "error": str(e),
                "message": "Failed to connect to Ollama. Make sure it's running."
            }
    
    def get_npc_response_streaming(
        self,
        npc_name: str,
        npc_personality: str,
        situation: str,
        player_action: str,
        context: Optional[str] = None,
        temperature: float = 0.7
    ):
        """
        Get an NPC response from Ollama with streaming output.
        
        Yields response text as it's generated.
        """
        prompt = self.create_npc_prompt(
            npc_name, npc_personality, situation, player_action, context
        )
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.api_endpoint, json=payload, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        yield chunk["response"]
        
        except requests.exceptions.RequestException as e:
            yield f"Error: {str(e)}"


if __name__ == "__main__":
    from pprint import pprint

    npc_dm = NPCDecisionMaker()
    npc_response = npc_dm.get_npc_response(
        npc_name="Test NPC",
        npc_personality="Test personality",
        situation="Test situation",
        player_action="Test action"
    )
    pprint(npc_response)
