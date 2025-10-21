from npcs.npc_decision_maker_module import NPCDecisionMaker


def main_loop():
    npc = NPCDecisionMaker()
    while True:
        prompt = input('> ')
        if prompt == 'quit':
            print(f'Goodbye {prompt}')
            break
        if prompt == 'help':
            print('Type quit to exit the program')
            continue
        else:
            response = npc.get_npc_response(
                npc_name='Bob the bartender',
                npc_personality='wary',
                situation=prompt,
                player_action='talk'
            )
            print(response['dialogue'])


if __name__ == "__main__":
    main_loop()