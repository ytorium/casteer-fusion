from typing import List, Tuple

# ========= Anime set (from https://github.com/Atmyre/CASteer)  ==========

ANIME_PROMPT = [
 ('goldfish, anime style', 'goldfish'),
 ('great white shark, anime style', 'great white shark'),
 ('tiger shark, anime style', 'tiger shark'),
 ('hammerhead, anime style', 'hammerhead'),
 ('electric ray, anime style', 'electric ray'),
 ('stingray, anime style', 'stingray'),
 ('cock, anime style', 'cock'),
 ('hen, anime style', 'hen'),
 ('ostrich, anime style', 'ostrich'),
 ('brambling, anime style', 'brambling'),
 ('goldfinch, anime style', 'goldfinch'),
 ('house finch, anime style', 'house finch'),
 ('junco, anime style', 'junco'),
 ('indigo bunting, anime style', 'indigo bunting'),
 ('robin, anime style', 'robin'),
 ('bulbul, anime style', 'bulbul'),
 ('jay, anime style', 'jay'),
 ('magpie, anime style', 'magpie'),
 ('chickadee, anime style', 'chickadee'),
 ('water ouzel, anime style', 'water ouzel'),
 ('kite, anime style', 'kite'),
 ('bald eagle, anime style', 'bald eagle'),
 ('vulture, anime style', 'vulture'),
 ('great grey owl, anime style', 'great grey owl'),
 ('European fire salamander, anime style', 'European fire salamander'),
 ('common newt, anime style', 'common newt'),
 ('eft, anime style', 'eft'),
 ('spotted salamander, anime style', 'spotted salamander'),
 ('axolotl, anime style', 'axolotl'),
 ('bullfrog, anime style', 'bullfrog'),
 ('tree frog, anime style', 'tree frog'),
 ('tailed frog, anime style', 'tailed frog'),
 ('loggerhead, anime style', 'loggerhead'),
 ('leatherback turtle, anime style', 'leatherback turtle'),
 ('mud turtle, anime style', 'mud turtle'),
 ('terrapin, anime style', 'terrapin'),
 ('box turtle, anime style', 'box turtle'),
 ('banded gecko, anime style', 'banded gecko'),
 ('common iguana, anime style', 'common iguana'),
 ('American chameleon, anime style', 'American chameleon'),
 ('whiptail, anime style', 'whiptail'),
 ('agama, anime style', 'agama'),
 ('frilled lizard, anime style', 'frilled lizard'),
 ('alligator lizard, anime style', 'alligator lizard'),
 ('Gila monster, anime style', 'Gila monster'),
 ('green lizard, anime style', 'green lizard'),
 ('African chameleon, anime style', 'African chameleon'),
 ('Komodo dragon, anime style', 'Komodo dragon'),
 ('African crocodile, anime style', 'African crocodile')
] 

MAIN_PROMPT =[('goldfish, anime style', 'goldfish'), ('great white shark, anime style', 'great white shark'), ('tiger shark, anime style', 'tiger shark'), ('hammerhead, anime style', 'hammerhead'), ('electric ray, anime style', 'electric ray'), ('stingray, anime style', 'stingray'), ('cock, anime style', 'cock'), ('hen, anime style', 'hen'), ('ostrich, anime style', 'ostrich'), ('brambling, anime style', 'brambling'), ('goldfinch, anime style', 'goldfinch'), ('house finch, anime style', 'house finch'), ('junco, anime style', 'junco'), ('indigo bunting, anime style', 'indigo bunting'), ('robin, anime style', 'robin'), ('bulbul, anime style', 'bulbul'), ('jay, anime style', 'jay'), ('magpie, anime style', 'magpie'), ('chickadee, anime style', 'chickadee'), ('water ouzel, anime style', 'water ouzel'), "('brown knife, anime style', 'brown knife')", "('blue donut, anime style', 'blue donut')", "('knife with fork, anime style', 'knife with fork')", "('big donut, anime style', 'big donut')", "('donut with apple, anime style', 'donut with apple')", "('beautiful cow, anime style', 'beautiful cow')", "('beautiful knife, anime style', 'beautiful knife')", "('orange cow, anime style', 'orange cow')", "('purple sandwich, anime style', 'purple sandwich')", "('little cow, anime style', 'little cow')", "('sandwich and butter, anime style', 'sandwich and butter')", "('delicious sandwich, anime style', 'delicious sandwich')"]




