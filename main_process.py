from npcs.npc_decision_maker_module import NPCDecisionMaker
from dotenv import load_dotenv
import random
import os

load_dotenv()
llm_url: str = os.getenv('OLLAMA_URL')
llm_model: str = os.getenv('OLLAMA_MODEL')
import json

def main_loop():
    npc: NPCDecisionMaker = NPCDecisionMaker(ollama_url=llm_url, model=llm_model)
    debug_mode: bool = False
    while True:
        prompt: str = input('> ')
        if prompt == 'quit':
            print(f'Goodbye {prompt}')
            break
        if prompt == 'help':
            print('Type quit to exit the program')
            continue
        if prompt == 'debug':
            debug_mode = not debug_mode
            print('Switching debug mode to', debug_mode)
        else:
            response: dict = npc.get_npc_response(
                npc_name='Bob the bartender',
                npc_personality=random.choice(['wary', 'cautious', 'inebriated', 'happy', 'buys', 'sad', 'bored', 'spiteful', 'rushed']),
                situation=prompt,
                player_action='talk'
            )

            print(f'With {response["emotion"]} the bartender says: {response["dialogue"]}')
            if debug_mode:
                print(f'Input: {prompt}')
                print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main_loop()