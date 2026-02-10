import random


class Player:
    def __init__(self, name):
        self.name = name
        self.health = 100
        self.inventory = []
        self.location = 'Entrance Hall'
        self.game_over = False

    def show_status(self):
        print(f"\n{self.name}, your current health is {self.health}.")
        print(f"Inventory: {', '.join(self.inventory) if self.inventory else 'empty'}")
        print(f"Current location: {self.location}\n")

    def pick_item(self, item):
        self.inventory.append(item)
        print(f"\nYou picked up {item}!\n")

    def take_damage(self, damage):
        self.health -= damage
        print(f"\nOh no! You lost {damage} health points.\n")
        if self.health <= 0:
            print(f"\n{self.name}, you've lost all your health and perished in the castle. Game Over.\n")
            self.game_over = True

    def heal(self, amount):
        self.health += amount
        print(f"\nYou healed {amount} health points! Your health is now {self.health}.\n")


class Castle:
    def __init__(self):
        self.rooms = {
            'Entrance Hall': {
                'description': "A vast hall with flickering torches lighting the stone walls. There are doors to the north and east. A magical glow surrounds you.",
                'actions': ['Go North', 'Go East', 'Examine Torches'],
                'items': [],
                'danger': None
            },
            'Library': {
                'description': "A dusty library with shelves full of old, forgotten books. There’s a glowing book on the table. A door to the west leads back.",
                'actions': ['Go West', 'Examine Book'],
                'items': ['Healing Potion'],
                'danger': None
            },
            'Armory': {
                'description': "The armory is filled with shiny weapons and armor. There is an exit to the south.",
                'actions': ['Go South', 'Take Sword', 'Examine Armor'],
                'items': ['Sword'],
                'danger': 'Trap'
            },
            'Throne Room': {
                'description': "A grand room with a high ceiling. A broken throne sits at the far end. Shadows creep along the walls.",
                'actions': ['Examine Throne', 'Go East'],
                'items': ['Shield'],
                'danger': 'Monster'
            },
            'Secret Passage': {
                'description': "A dark and narrow passage leading to the unknown. You can feel danger nearby.",
                'actions': ['Go West', 'Proceed to the End'],
                'items': [],
                'danger': 'Monster'
            }
        }

    def get_room_actions(self, room):
        return self.rooms[room]['actions']

    def describe_room(self, room):
        return self.rooms[room]['description']

    def get_room_items(self, room):
        return self.rooms[room]['items']

    def get_room_danger(self, room):
        return self.rooms[room]['danger']


def start_game():
    print("Welcome to the Mysterious Castle Adventure!")
    name = input("Enter your character's name: ")
    player = Player(name)
    castle = Castle()

    print(f"\n{player.name}, you find yourself in front of a grand and eerie castle. Are you brave enough to enter?")

    while not player.game_over:
        player.show_status()
        room_description = castle.describe_room(player.location)
        print(room_description)

        # Check for dangers in the room
        danger = castle.get_room_danger(player.location)
        if danger:
            handle_danger(player, danger)
            if player.game_over:
                break

        # Display available actions
        actions = castle.get_room_actions(player.location)
        print("What would you like to do?")
        for idx, action in enumerate(actions, start=1):
            print(f"{idx}. {action}")

        try:
            choice = int(input("Choose an action (1/2/3...): "))
            action = actions[choice - 1]
            process_action(player, castle, action)
        except (ValueError, IndexError):
            print("\nInvalid choice! Please select a valid action.\n")

        if player.health <= 0:
            print(f"\n{player.name}, you have perished in the castle. Game Over.")
            player.game_over = True
        if player.location == 'Secret Passage' and 'Proceed to the End' in actions:
            print(f"\nCongratulations, {player.name}! You have successfully navigated the castle and escaped alive!")
            player.game_over = True


def handle_danger(player, danger):
    if danger == 'Trap':
        print("\nYou triggered a trap! Arrows shoot from the walls!")
        player.take_damage(20)
    elif danger == 'Monster':
        print("\nA ferocious monster appears!")
        if 'Sword' in player.inventory:
            print("You bravely fight the monster with your sword and defeat it!")
        else:
            print("You have no weapon to defend yourself!")
            player.take_damage(30)


def process_action(player, castle, action):
    if action == 'Go North':
        player.location = 'Library'
        print("\nYou head north and enter the Library.\n")
    elif action == 'Go East':
        if player.location == 'Entrance Hall':
            player.location = 'Armory'
            print("\nYou go east and find yourself in the Armory.\n")
        elif player.location == 'Throne Room':
            player.location = 'Secret Passage'
            print("\nYou find a secret passage leading to darkness...\n")
    elif action == 'Go West':
        if player.location == 'Library':
            player.location = 'Entrance Hall'
            print("\nYou head west and return to the Entrance Hall.\n")
        elif player.location == 'Secret Passage':
            player.location = 'Throne Room'
            print("\nYou head west and return to the Throne Room.\n")
    elif action == 'Go South':
        player.location = 'Entrance Hall'
        print("\nYou head south and return to the Entrance Hall.\n")
    elif action == 'Examine Torches':
        print("\nThe torches are ancient, but they burn brightly. One torch looks a bit loose.")
        if input("Do you want to pull the loose torch? (yes/no): ").lower() == 'yes':
            print("The wall slides open, revealing a secret passage to the Throne Room!\n")
            player.location = 'Throne Room'
    elif action == 'Examine Book':
        print("\nThe book is old and mysterious. You feel compelled to read it.")
        if input("Do you want to open the book? (yes/no): ").lower() == 'yes':
            outcome = random.choice([True, False])
            if outcome:
                print("The book grants you knowledge of the castle's secret paths!\n")
            else:
                print("A dark curse falls upon you. You lose 20 health points!\n")
                player.take_damage(20)
    elif action == 'Take Sword':
        if 'Sword' not in player.inventory:
            player.pick_item('Sword')
        else:
            print("\nYou already have the sword.\n")
    elif action == 'Examine Armor':
        print("\nThe armor is rusty and useless, but one of the helmets hides a small key.")
        if input("Do you want to take the key? (yes/no): ").lower() == 'yes':
            player.pick_item('Key')
    elif action == 'Examine Throne':
        print("\nThe throne is cracked and worn. It doesn't seem like anyone has sat here for centuries.")
        if 'Key' in player.inventory:
            print("You use the key to open a hidden compartment beneath the throne. It reveals a secret passage!")
        else:
            print("There seems to be a hidden compartment, but it's locked.\n")
    elif action == 'Proceed to the End':
        player.game_over = True
    else:
        print("\nInvalid action. Try again.\n")


# Start the game
if __name__ == "__main__":
    start_game()
