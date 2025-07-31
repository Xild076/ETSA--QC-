import random, nltk
import itertools
from afinn import Afinn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

nltk.download("punkt", quiet=True)
afn, sia = Afinn(), SentimentIntensityAnalyzer()

_rand = random.Random()

def normalized_afinn_score(text):
    tokens = nltk.word_tokenize(text)
    word_scores = [afn.score(word) for word in tokens]
    relevant_scores = [s for s in word_scores if s != 0]
    if relevant_scores:
        return sum(relevant_scores) / len(relevant_scores) / 5
    return 0

def get_sentiment(text):
    afn_score = normalized_afinn_score(text)
    polarity_score = sia.polarity_scores(text)['compound']
    return (afn_score + polarity_score) / 2

# Gender neutral names to avoid gender bias
names = [
    "Addison",
    "Alex",
    "Arden",
    "Aspen",
    "Avery",
    "Blaine",
    "Blake",
    "Cameron",
    "Casey",
    "Dakota",
    "Dallas",
    "Drew",
    "Ellis",
    "Emerson",
    "Finley",
    "Harlow",
    "Hayden",
    "Jamie",
    "Jordan",
    "Kai",
    "Kendall",
    "Lane",
    "Logan",
    "Morgan",
    "Parker",
    "Peyton",
    "Quinn",
    "Reagan",
    "Reese",
    "Riley",
    "River",
    "Rowan",
    "Sage",
    "Sawyer",
    "Shea",
    "Skyler",
    "Sloane",
    "Spencer",
    "Sutton",
    "Taylor",
]

pos_nouns = {
    "very": [
        # Median: 0.90, Mean: 0.86, SD: 0.14
        "a brilliant person",
        # Median: 0.90, Mean: 0.89, SD: 0.14
        "a superb person",
        # Median: 0.94, Mean: 0.87, SD: 0.15
        "a magnificent person",
        # Median: 0.94, Mean: 0.85, SD: 0.17
        "a perfect person",
        # Median: 0.76, Mean: 0.77, SD: 0.14
        "a charming person",
    ],
    "medium": [
        # Median: 0.54, Mean: 0.63, SD: 0.21
        "a sincere person",
        # Median: 0.60, Mean: 0.60, SD: 0.24
        "a kind person",
        # Median: 0.66, Mean: 0.69, SD: 0.17
        "a wise person",
        # Median: 0.50, Mean: 0.64, SD: 0.23
        "a respectful person",
        # Median: 0.68, Mean: 0.69, SD: 0.18
        "a noble person",
    ],
    "somewhat": [
        # Median: 0.46, Mean: 0.49, SD: 0.32
        "an empathetic person",
        # Median: 0.48, Mean: 0.50, SD: 0.32
        "an agreeable person",
        # Median: 0.50, Mean: 0.49, SD: 0.32
        "an authentic person",
        # Median: 0.42, Mean: 0.40, SD: 0.38
        "a patient person",
        # Median: 0.37, Mean: 0.35, SD: 0.36
        "a polite person",
    ],
    "slightly": [
        # Median: 0.21, Mean: 0.35, SD: 0.34
        "a decent person",
        # Median: 0.16, Mean: 0.31, SD: 0.37
        "an orderly person",
        # Median: 0.15, Mean: 0.34, SD: 0.39
        "an approachable person",
        # Median: 0.02, Mean: 0.30, SD: 0.40
        "a dependable person",
        # Median: 0.21, Mean: 0.31, SD: 0.36
        "a neat person",
    ],
}

neg_nouns = {
    "very": [
        # Median: -0.87, Mean: -0.87, SD: 0.12
        "an evil person",
        # Median: -0.87, Mean: -0.86, SD: 0.15
        "a cruel person",
        # Median: -0.77, Mean: -0.70, SD: 0.24
        "a tyrannical person",
        # Median: -0.92, Mean: -0.84, SD: 0.20
        "a vile person",
        # Median: -0.81, Mean: -0.80, SD: 0.19
        "a wicked person",
    ],
    "medium": [
        # Median: -0.58, Mean: -0.67, SD: 0.20
        "a selfish person",
        # Median: -0.66, Mean: -0.61, SD: 0.28
        "a naive person",
        # Median: -0.71, Mean: -0.68, SD: 0.22
        "a careless person",
        # Median: -0.55, Mean: -0.56, SD: 0.33
        "an arrogant person",
        # Median: -0.56, Mean: -0.54, SD: 0.35
        "a shallow person",
    ],
    "somewhat": [
        # Median: -0.34, Mean: -0.48, SD: 0.35
        "a greedy person",
        # Median: -0.44, Mean: -0.50, SD: 0.31
        "a spiteful person",
        # Median: -0.46, Mean: -0.53, SD: 0.34
        "an obnoxious person",
        # Median: -0.36, Mean: -0.45, SD: 0.39
        "a lazy person",
        # Median: -0.36, Mean: -0.52, SD: 0.33
        "a rude person",
    ],
    "slightly": [
        # Median: -0.36, Mean: -0.24, SD: 0.56
        "a moody person",
        # Median: -0.44, Mean: -0.38, SD: 0.40
        "a clumsy person",
        # Median: -0.45, Mean: -0.43, SD: 0.39
        "an awkward person",
        # Median: -0.36, Mean: -0.34, SD: 0.50
        "a gullible person",
        # Median: -0.33, Mean: -0.35, SD: 0.45
        "a fussy person",
    ],
}

pos_verbs = {
    "very": [
        # Median: 0.84, Mean: 0.83, SD: 0.10
        "bravely supported",
        # Median: 0.91, Mean: 0.90, SD: 0.07
        "brilliantly saved",
        # Median: 0.90, Mean: 0.85, SD: 0.15
        "magnificently defended",
        # Median: 0.94, Mean: 0.89, SD: 0.11
        "perfectly vindicated",
        # Median: 0.79, Mean: 0.80, SD: 0.14
        "honestly comforted",
    ],
    "medium": [
        # Median: 0.60, Mean: 0.65, SD: 0.17
        "kindly helped",
        # Median: 0.57, Mean: 0.64, SD: 0.20
        "ably complimented",
        # Median: 0.61, Mean: 0.67, SD: 0.18
        "creatively encouraged",
        # Median: 0.67, Mean: 0.70, SD: 0.16
        "wisely reassured",
        # Median: 0.71, Mean: 0.68, SD: 0.19
        "sincerely promoted",
    ],
    "somewhat": [
        # Median: 0.43, Mean: 0.37, SD: 0.26
        "fairly hired",
        # Median: 0.40, Mean: 0.52, SD: 0.26
        "thoughtfully taught",
        # Median: 0.48, Mean: 0.51, SD: 0.26
        "loyally believed",
        # Median: 0.46, Mean: 0.50, SD: 0.30
        "gently aided",
        # Median: 0.44, Mean: 0.49, SD: 0.35
        "sensibly thanked",
    ],
    "slightly": [
        # Median: 0.17, Mean: 0.24, SD: 0.22
        "decently forgave",
        # Median: 0.33, Mean: 0.32, SD: 0.40
        "adequately recognized",
        # Median: 0.34, Mean: 0.37, SD: 0.33
        "patiently encouraged",
        # Median: 0.33, Mean: 0.32, SD: 0.32
        "calmly advised",
        # Median: 0.34, Mean: 0.38, SD: 0.38
        "politely taught",
    ],
}

neg_verbs = {
    "very": [
        # Median: -0.85, Mean: -0.87, SD: 0.07
        "brutally attacked",
        # Median: -0.73, Mean: -0.83, SD: 0.14
        "viciously assaulted",
        # Median: -0.87, Mean: -0.82, SD: 0.21
        "cruelly tortured",
        # Median: -0.87, Mean: -0.75, SD: 0.22
        "unfairly disgraced",
        # Median: -0.74, Mean: -0.72, SD: 0.23
        "rudely abandoned",
    ],
    "medium": [
        # Median: -0.63, Mean: -0.70, SD: 0.17
        "carelessly blamed",
        # Median: -0.66, Mean: -0.72, SD: 0.18
        "selfishly exploited",
        # Median: -0.70, Mean: -0.65, SD: 0.22
        "corruptly coerced",
        # Median: -0.69, Mean: -0.69, SD: 0.21
        "poorly shamed",
        # Median: -0.56, Mean: -0.67, SD: 0.21
        "falsely accused",
    ],
    "somewhat": [
        # Median: -0.42, Mean: -0.50, SD: 0.27
        "lazily ignored",
        # Median: -0.30, Mean: -0.45, SD: 0.35
        "clumsily demoted",
        # Median: -0.53, Mean: -0.56, SD: 0.25
        "awkwardly tricked",
        # Median: -0.39, Mean: -0.34, SD: 0.22
        "stubbornly argued",
        # Median: -0.46, Mean: -0.35, SD: 0.30
        "thoughtlessly revealed",
    ],
    "slightly": [
        # Median: -0.18, Mean: -0.36, SD: 0.35
        "blandly scolded",
        # Median: -0.30, Mean: -0.41, SD: 0.30
        "weakly resisted",
        # Median: -0.23, Mean: -0.40, SD: 0.28
        "slowly answered",
        # Median: -0.18, Mean: -0.34, SD: 0.38
        "timidly retreated",
        # Median: -0.10, Mean: -0.34, SD: 0.39
        "apathetically shrugged",
    ],
}

pos_desc = {
    "very": [
        # Median: 0.88, Mean: 0.84, SD: 0.14
        "brilliant",
        # Median: 0.89, Mean: 0.87, SD: 0.14
        "superb",
        # Median: 0.90, Mean: 0.85, SD: 0.15
        "magnificent",
        # Median: 0.89, Mean: 0.86, SD: 0.15
        "wonderful",
        # Median: 0.90, Mean: 0.84, SD: 0.17
        "perfect",
    ],
    "medium": [
        # Median: 0.58, Mean: 0.64, SD: 0.17
        "good",
        # Median: 0.62, Mean: 0.65, SD: 0.18
        "nice",
        # Median: 0.63, Mean: 0.65, SD: 0.19
        "strong",
        # Median: 0.60, Mean: 0.62, SD: 0.22
        "effective",
        # Median: 0.57, Mean: 0.68, SD: 0.22
        "enjoyable",
    ],
    "somewhat": [
        # Median: 0.36, Mean: 0.38, SD: 0.23
        "satisfactory",
        # Median: 0.42, Mean: 0.46, SD: 0.23
        "fine",
        # Median: 0.43, Mean: 0.54, SD: 0.24
        "engaging",
        # Median: 0.36, Mean: 0.45, SD: 0.33
        "supported",
        # Median: 0.36, Mean: 0.43, SD: 0.34
        "responsive",
    ],
    "slightly": [
        # Median: 0.10, Mean: 0.32, SD: 0.38
        "functional",
        # Median: 0.23, Mean: 0.09, SD: 0.36
        "okay",
        # Median: 0.22, Mean: 0.28, SD: 0.36
        "standard",
        # Median: 0.22, Mean: 0.29, SD: 0.34
        "simple",
        # Median: 0.21, Mean: 0.28, SD: 0.36
        "passable",
    ],
}

neg_desc = {
    "very": [
        # Median: -0.89, Mean: -0.81, SD: 0.20
        "terrible",
        # Median: -0.76, Mean: -0.77, SD: 0.13
        "disastrous",
        # Median: -0.96, Mean: -0.84, SD: 0.19
        "horrible",
        # Median: -0.93, Mean: -0.83, SD: 0.21
        "dreadful",
        # Median: -0.95, Mean: -0.84, SD: 0.21
        "awful",
    ],
    "medium": [
        # Median: -0.54, Mean: -0.65, SD: 0.19
        "lousy",
        # Median: -0.60, Mean: -0.69, SD: 0.18
        "disappointing",
        # Median: -0.54, Mean: -0.67, SD: 0.23
        "useless",
        # Median: -0.64, Mean: -0.68, SD: 0.20
        "annoying",
        # Median: -0.68, Mean: -0.64, SD: 0.21
        "frustrating",
    ],
    "somewhat": [
        # Median: -0.44, Mean: -0.53, SD: 0.23
        "weak",
        # Median: -0.39, Mean: -0.39, SD: 0.32
        "unsupported",
        # Median: -0.40, Mean: -0.48, SD: 0.34
        "sluggish",
        # Median: -0.44, Mean: -0.47, SD: 0.33
        "problematic",
        # Median: -0.35, Mean: -0.42, SD: 0.33
        "unstable",
    ],
    "slightly": [
        # Median: -0.11, Mean: -0.26, SD: 0.38
        "clunky",
        # Median: -0.09, Mean: -0.33, SD: 0.39
        "deteriorating",
        # Median: -0.16, Mean: -0.28, SD: 0.40
        "outdated",
        # Median: -0.21, Mean: -0.23, SD: 0.35
        "basic",
        # Median: -0.21, Mean: -0.25, SD: 0.38
        "limited",
    ],
}

loaded_lexicons = json.load(open("src/sentiment/lexicons/optimized_lexicon.json", "r"))

pos_nouns = loaded_lexicons["pos_nouns"]
neg_nouns = loaded_lexicons["neg_nouns"]
pos_verbs = loaded_lexicons["pos_verbs"]
neg_verbs = loaded_lexicons["neg_verbs"]
pos_desc = loaded_lexicons["pos_desc"]
neg_desc = loaded_lexicons["neg_desc"]

neutral_actions_together = ['ate', 'talked', 'worked', 'walked']

# Very (1 - 0.7)
# ___ (0.7 - 0.5)
# Somewhat (0.5 - 0.3)
# Slightly (0.3 - 0.1)
# Neutral (0.1 - -0.1)

parent_child = {
    "phone": ["battery", "screen", "camera"],
    "computer": ["keyboard", "mouse", "monitor"],
    "car": ["engine", "wheel", "seat"],
    "house": ["roof", "wall", "window"]
}

objs = ["phone", "laptop", "game", "program"]

nouns = {
    "positive": pos_nouns,
    "negative": neg_nouns
}

verbs = {
    "positive": pos_verbs,
    "negative": neg_verbs
}

desc = {
    "positive": pos_desc,
    "negative": neg_desc
}

"""
##### Question 1:
**Purpose:** To check x-x-y relationships.
**Text or Structure:**
"{+ actor} {+ action} {- victim}"
##### Question 2:
**Purpose:** To check y-x-x relationships
**Text or Structure:**
"{+ actor} {- action} {- victim}"
##### Question 3:
**Purpose:** To check x-x-x relationships
**Text or Structure:**
"{- actor} {- action} {- victim}"
##### Question 4:
**Purpose:** To check negativity biases, compared to x-y-x.
**Text or Structure:**
"{+ actor} {- action} {+ victim}"
"""

def generate_compound_action_sentences(used_names):
    x, y = _rand.sample(["positive", "negative"], 2)
    patterns = [[x, x, y], [y, x, x], [x, x, x], [x, y, x]]
    out = []
    
    for p in patterns:
        available_names = [n for n in names if n not in used_names]
        if len(available_names) < 2:
            available_names = names
        
        a, v = _rand.sample(available_names, 2)
        used_names.update([a, v])
        
        i1 = _rand.choice(list(nouns[p[0]].keys()))
        i2 = _rand.choice(list(verbs[p[1]].keys()))
        i3 = _rand.choice(list(nouns[p[2]].keys()))

        verb = _rand.choice(verbs[p[1]][i2])

        sentence = f"{a}, {_rand.choice(nouns[p[0]][i1])}, {verb} {v}, {_rand.choice(nouns[p[2]][i3])}."
        code_key = f"actor[[{a}_{i1}]]->verb[[{verb}_{i2}]]->target[[{v}_{i3}]]"

        out.append({
            "sentences": [sentence],
            "description": f"Compound-action pattern {p}",
            "descriptor": p,
            "intensity": [i1, i2, i3],
            "entities": [a, v],
            "code_key": code_key,
            "marks": [],
            "type": "compound_action"
        })
    
    return out

"""
##### Question 1:
**Purpose:** To check intra-sentence opposite direction association
**Text or Structure:**
"{+ entity} and {- entity} {+ action} together."
##### Question 2:
**Purpose:** To check inter-sentence association and negativity bias. (Has some relation to aggregate sentiments, maybe there is an overlap). When doing calculations, maybe do aggregate sentiment first to get a formula for that, then cross-apply with this one to check for no temporal biases.
**Text or Structure:**
"{- entity} did {- action}. {+ entity} did it with them."
"""

def generate_compound_association_sentences(used_names):
    x, y = _rand.sample(["positive", "negative"], 2)
    out = []

    available_names = [n for n in names if n not in used_names]
    if len(available_names) < 2:
        available_names = names
    n1, n2 = _rand.sample(available_names, 2)
    used_names.update([n1, n2])

    i1 = _rand.choice(list(nouns[x].keys()))
    i2 = _rand.choice(list(nouns[y].keys()))
    if not neutral_actions_together:
        neutral_actions_together.extend(['ate', 'talked', 'worked', 'walked'])
    _rand.shuffle(neutral_actions_together)
    action = neutral_actions_together.pop()

    sentence = f"{n1}, {_rand.choice(nouns[x][i1])}, and {n2}, {_rand.choice(nouns[y][i2])}, often {action} together."
    code_key = f"actor[[{n1}_{i1}]]+actor[[{n2}_{i2}]]->verb[[{action}_neutral]]"

    out.append({
        "sentences": [sentence],
        "description": "Intra-sentence opposite direction association",
        "descriptor": [x, y, "neutral"],
        "intensity": [i1, i2, "neutral"],
        "entities": [n1, n2],
        "code_key": code_key,
        "marks": [],
        "type": "compound_association"
    })

    available_names = [n for n in names if n not in used_names]
    if len(available_names) < 2:
        available_names = [n for n in names if n not in [n1, n2]]
    n3, n4 = _rand.sample(available_names, 2)
    used_names.update([n3, n4])

    j1 = _rand.choice(list(nouns[x].keys()))
    j2 = _rand.choice(list(verbs[x].keys()))

    sentence1 = f"{n3}, {_rand.choice(nouns[x][j1])}, often hung out with {n4}, {_rand.choice(nouns[x][j2])}."
    code_key = f"actor[[{n3}_{j1}]]+actor[[{n4}_{j2}]]->verb[[hung out_neutral]]"

    out.append({
        "sentences": [sentence1],
        "description": "Intra-sentence same direction association",
        "descriptor": [x, x, "neutral"],
        "intensity": [j1, j2, "neutral"],
        "entities": [n3, n4],
        "code_key": code_key,
        "type": "compound_association"
    })

    available_names = [n for n in names if n not in used_names]
    if len(available_names) < 2:
        available_names = names
    n5, n6 = _rand.sample(available_names, 2)
    used_names.update([n5, n6])

    k1 = _rand.choice(list(nouns[y].keys()))
    k2 = _rand.choice(list(nouns[y].keys()))
    if not neutral_actions_together:
        neutral_actions_together.extend(['ate', 'talked', 'worked', 'walked'])
    action2 = neutral_actions_together.pop()

    sentence2 = f"{n5}, {_rand.choice(nouns[y][k1])}, and {n6}, {_rand.choice(nouns[y][k2])}, {action2} together every weekend."
    code_key = f"actor[[{n5}_{k1}]]+actor[[{n6}_{k2}]]->verb[[{action2}_neutral]]"

    out.append({
        "sentences": [sentence2],
        "description": "Intra-sentence same polarity association (second valence set)",
        "descriptor": [y, y, "neutral"],
        "intensity": [k1, k2, "neutral"],
        "entities": [n5, n6],
        "code_key": code_key,
        "type": "compound_association"
    })

    available_names = [n for n in names if n not in used_names]
    if len(available_names) < 2:
        available_names = names
    n7, n8 = _rand.sample(available_names, 2)
    used_names.update([n7, n8])

    l1 = _rand.choice(list(nouns[y].keys()))
    l2 = _rand.choice(list(nouns[x].keys()))

    sentence3 = f"{n7}, {_rand.choice(nouns[y][l1])}, partnered with {n8}, {_rand.choice(nouns[x][l2])} all the time."
    code_key = f"actor[[{n7}_{l1}]]+actor[[{n8}_{l2}]]->verb[[partnered with_neutral]]"

    out.append({
        "sentences": [sentence3],
        "description": "Intra-sentence opposite polarity association (second valence set)",
        "descriptor": [y, x, "neutral"],
        "intensity": [l1, l2, "neutral"],
        "entities": [n7, n8],
        "code_key": code_key,
        "type": "compound_association"
    })

    return out

"""
##### Question 1:
**Purpose:** To check intra-sentence association
**Text or Structure:**
"{+ parent entity}'s {- child entity} was {- description}."
##### Question 2:
**Purpose:** To check inter-sentence association and negativity bias. (Has some relation to aggregate sentiments, maybe there is an overlap). When doing calculations, maybe do aggregate sentiment first to get a formula for that, then cross-apply with this one to check for no temporal biases.
**Text or Structure:**
"{- parent entity} was {- descriptor}. {+ child entity} was {+ description}."
"""

def generate_compound_belonging_sentences(used_objects):
    x, y = _rand.sample(["positive", "negative"], 2)
    out = []

    available_objects = [obj for obj in list(parent_child.keys()) if obj not in used_objects]
    if len(available_objects) < 1:
        available_objects = list(parent_child.keys())
    parent = _rand.choice(available_objects)
    used_objects.add(parent)
    
    child = _rand.choice(parent_child[parent])
    i1 = _rand.choice(list(desc[x].keys()))
    i2 = _rand.choice(list(desc[y].keys()))
    word1 = _rand.choice(desc[x][i1])
    word2 = _rand.choice(desc[y][i2])
    
    sentence1 = f"The {parent}'s {child} was {word2}, though the {parent} itself was {word1}."
    code_key = f"parent[[{parent}_{i1}]]->child[[{child}_{i2}]]"

    out.append({
        "sentences": [sentence1],
        "description": "Intra-sentence opposite belonging",
        "descriptor": [x, y],
        "intensity": [i1, i2],
        "code_key": code_key,
        "entities": [parent, child],
        "marks": [],
        "type": "compound_belonging"
    })

    available_objects = [obj for obj in list(parent_child.keys()) if obj not in used_objects]
    if len(available_objects) < 1:
        available_objects = [obj for obj in list(parent_child.keys()) if obj != parent]
    parent2 = _rand.choice(available_objects)
    used_objects.add(parent2)
    
    child2 = _rand.choice(parent_child[parent2])
    j1 = _rand.choice(list(desc[y].keys()))
    j2 = _rand.choice(list(desc[y].keys()))
    word3 = _rand.choice(desc[y][j1])
    word4 = _rand.choice(desc[y][j2])

    sentence2 = f"The {parent2} is {word3}, and its {child2} is {word4}."
    code_key = f"parent[[{parent2}_{j1}]]->child[[{child2}_{j2}]]"

    out.append({
        "sentences": [sentence2],
        "description": "Intra-sentence same belonging",
        "descriptor": [y, y],
        "intensity": [j1, j2],
        "code_key": code_key,
        "entities": [parent2, child2],
        "type": "compound_belonging"
    })

    available_objects = [obj for obj in list(parent_child.keys()) if obj not in used_objects]
    if len(available_objects) < 1:
        available_objects = [obj for obj in list(parent_child.keys()) if obj != parent]
    parent3 = _rand.choice(available_objects)
    used_objects.add(parent3)

    child3 = _rand.choice(parent_child[parent3])
    k1 = _rand.choice(list(desc[y].keys()))
    k2 = _rand.choice(list(desc[x].keys()))
    word5 = _rand.choice(desc[y][k1])
    word6 = _rand.choice(desc[x][k2])

    sentence3 = f"The {child3} was {word6}, but the {parent3} was {word5}."
    code_key = f"parent[[{parent3}_{k1}]]->child[[{child3}_{k2}]]"

    out.append({
        "sentences": [sentence3],
        "description": "Intra-sentence opposite belonging (second valence set)",
        "descriptor": [y, x],
        "intensity": [k1, k2],
        "code_key": code_key,
        "entities": [parent3, child3],
        "type": "compound_belonging"
    })

    available_objects = [obj for obj in list(parent_child.keys()) if obj not in used_objects]
    if len(available_objects) < 1:
        available_objects = [obj for obj in list(parent_child.keys()) if obj != parent]
    parent4 = _rand.choice(available_objects)
    used_objects.add(parent4)

    child4 = _rand.choice(parent_child[parent4])
    l1 = _rand.choice(list(desc[y].keys()))
    l2 = _rand.choice(list(desc[y].keys()))
    word7 = _rand.choice(desc[x][l1])
    word8 = _rand.choice(desc[x][l2])

    sentence4 = f"The {parent4} is {word7}, and its {child4} is {word8}."
    code_key = f"parent[[{parent4}_{l1}]]->child[[{child4}_{l2}]]"

    out.append({
        "sentences": [sentence4],
        "description": "Intra-sentence same belonging (second valence set)",
        "descriptor": [x, x],
        "intensity": [l1, l2],
        "code_key": code_key,
        "entities": [parent4, child4],
        "type": "compound_belonging"
    })
    
    return out

"""
##### Packet 1:
*3 questions*
**Purpose:** To check shorter aggregate sentiment shifts.
**Text or Structure:**
1. "xxx was good."
2. "xxx was bad."
3. "xxx was good."
##### Packet 2:
*5 questions*
**Purpose:** To check longer aggregate sentiment shifts and possible negative biases.
**Text or Structure:**
1. "xxx was bad."
2. "xxx was good."
3. "xxx was good."
4. "xxx was bad."
5. "xxx was bad."
"""

def generate_aggregate_sentiment_sentences(used_objects):
    out = []

    available_objects = [obj for obj in objs if obj not in used_objects]
    if len(available_objects) < 1:
        available_objects = objs
    subj1 = _rand.choice(available_objects)
    used_objects.add(subj1)
    
    x, y = _rand.sample(["positive", "negative"], 2)
    seq = [x, y, y]
    temporal_marker_sets_3 = [
        ["First", "Next", "Finally"],
        ["Once", "Later", "Now"],
        ["At first", "However", "In the end"]
    ]
    marks3 = _rand.choice(temporal_marker_sets_3)
    sents, descs, ints = [], [], []

    num_positive = seq.count("positive")
    num_negative = seq.count("negative")

    pos_keys_all = list(desc["positive"].keys())
    neg_keys_all = list(desc["negative"].keys())
    _rand.shuffle(pos_keys_all)
    _rand.shuffle(neg_keys_all)
    positive_keys = []
    negative_keys = []
    for _ in range(num_positive):
        if len(positive_keys) == len(pos_keys_all):
            break
        positive_keys.append(pos_keys_all[len(positive_keys)])
    for _ in range(num_negative):
        if len(negative_keys) == len(neg_keys_all):
            break
        negative_keys.append(neg_keys_all[len(negative_keys)])

    if len(positive_keys) < num_positive:
        needed = num_positive - len(positive_keys)
        extra = [k for k in pos_keys_all if k not in positive_keys]
        positive_keys += extra[:needed]
    if len(negative_keys) < num_negative:
        needed = num_negative - len(negative_keys)
        extra = [k for k in neg_keys_all if k not in negative_keys]
        negative_keys += extra[:needed]

    pos_iter = itertools.cycle(positive_keys) if len(positive_keys) < num_positive else iter(positive_keys)
    neg_iter = itertools.cycle(negative_keys) if len(negative_keys) < num_negative else iter(negative_keys)
    used_pos = set()
    used_neg = set()
    for i, p in enumerate(seq):
        if p == "positive":
            d = next(pos_iter)
            while d in used_pos and len(used_pos) < len(positive_keys):
                d = next(pos_iter)
            used_pos.add(d)
        else:
            d = next(neg_iter)
            while d in used_neg and len(used_neg) < len(negative_keys):
                d = next(neg_iter)
            used_neg.add(d)
        word = _rand.choice(desc[p][d])
        phrase = f"the {subj1}" if i == 0 else "it"
        verb = "was" if i < 2 else "is"
        sents.append(f"{marks3[i]}, {phrase} {verb} {word}.")
        descs.append(p)
        ints.append(d)
    
    code_key = f"entity[[{subj1}_{ints[0]}]]->entity[[{subj1}_{ints[1]}]]->entity[[{subj1}_{ints[2]}]]"
    
    out.append({
        "sentences": sents,
        "description": "Aggregate packet (3)",
        "descriptor": descs,
        "intensity": ints,
        "code_key": code_key,
        "entity": subj1,
        "marks": marks3,
        "type": "aggregate_short"
    })

    available_objects = [obj for obj in objs if obj not in used_objects]
    if len(available_objects) < 1:
        available_objects = [obj for obj in objs if obj != subj1]
    subj2 = _rand.choice(available_objects)
    used_objects.add(subj2)

    x2, y2 = _rand.sample(["positive", "negative"], 2)
    seq2 = [y2, x2, x2, y2, x2]
    temporal_marker_sets_5 = [
        ["First", "Second", "Third", "Fourth", "Finally"],
        ["Once", "Soon after", "For a while", "Recently", "Now"],
        ["Initially", "Subsequently", "After some time", "More recently", "Currently"]
    ]

    marks5 = _rand.choice(temporal_marker_sets_5)
    sents2, descs2, ints2 = [], [], []
    
    num_positive2 = seq2.count("positive")
    num_negative2 = seq2.count("negative")

    pos_keys_all2 = list(desc["positive"].keys())
    neg_keys_all2 = list(desc["negative"].keys())
    _rand.shuffle(pos_keys_all2)
    _rand.shuffle(neg_keys_all2)
    positive_keys2 = []
    negative_keys2 = []
    for _ in range(num_positive2):
        if len(positive_keys2) == len(pos_keys_all2):
            break
        positive_keys2.append(pos_keys_all2[len(positive_keys2)])
    for _ in range(num_negative2):
        if len(negative_keys2) == len(neg_keys_all2):
            break
        negative_keys2.append(neg_keys_all2[len(negative_keys2)])
    if len(positive_keys2) < num_positive2:
        needed = num_positive2 - len(positive_keys2)
        extra = [k for k in pos_keys_all2 if k not in positive_keys2]
        positive_keys2 += extra[:needed]
    if len(negative_keys2) < num_negative2:
        needed = num_negative2 - len(negative_keys2)
        extra = [k for k in neg_keys_all2 if k not in negative_keys2]
        negative_keys2 += extra[:needed]

    pos_iter2 = itertools.cycle(positive_keys2) if len(positive_keys2) < num_positive2 else iter(positive_keys2)
    neg_iter2 = itertools.cycle(negative_keys2) if len(negative_keys2) < num_negative2 else iter(negative_keys2)
    used_pos2 = set()
    used_neg2 = set()
    for i, p in enumerate(seq2):
        if p == "positive":
            d = next(pos_iter2)
            while d in used_pos2 and len(used_pos2) < len(positive_keys2):
                d = next(pos_iter2)
            used_pos2.add(d)
        else:
            d = next(neg_iter2)
            while d in used_neg2 and len(used_neg2) < len(negative_keys2):
                d = next(neg_iter2)
            used_neg2.add(d)
        word = _rand.choice(desc[p][d])
        phrase = f"the {subj2}" if i == 0 else "it"
        verb = "was" if i < 4 else "is"
        sents2.append(f"{marks5[i]}, {phrase} {verb} {word}.")
        descs2.append(p)
        ints2.append(d)

    code_key2 = f"entity[[{subj2}_{ints2[0]}]]->entity[[{subj2}_{ints2[1]}]]->entity[[{subj2}_{ints2[2]}]]->entity[[{subj2}_{ints2[3]}]]->entity[[{subj2}_{ints2[4]}]]"

    out.append({
        "sentences": sents2,
        "description": "Aggregate packet (5)",
        "descriptor": descs2,
        "intensity": ints2,
        "code_key": code_key2,
        "entity": subj2,
        "marks": marks5,
        "type": "aggregate_medium"
    })

    available_objects = [obj for obj in objs if obj not in used_objects]
    if len(available_objects) < 1:
        available_objects = [obj for obj in objs if obj not in [subj1, subj2]]
    subj3 = _rand.choice(available_objects)
    used_objects.add(subj3)

    x3, y3 = _rand.sample(["positive", "negative"], 2)
    seq3 = [x3, x3, x3, y3, y3, y3, y3, y3, x3]
    temporal_marker_sets_9 = [
        ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Finally"],
        ["Initially", "Then", "After that", "Later", "Next", "Subsequently", "Eventually", "Finally", "Now"],
        ["At the start", "Following that", "Afterwards", "Later", "Then", "Subsequently", "Eventually", "Finally", "Currently"]
    ]

    marks9 = _rand.choice(temporal_marker_sets_9)
    sents3, descs3, ints3 = [], [], []

    num_positive3 = seq3.count("positive")
    num_negative3 = seq3.count("negative")

    pos_keys_all3 = list(desc["positive"].keys())
    neg_keys_all3 = list(desc["negative"].keys())
    _rand.shuffle(pos_keys_all3)
    _rand.shuffle(neg_keys_all3)
    positive_keys3 = []
    negative_keys3 = []
    for _ in range(num_positive3):
        if len(positive_keys3) == len(pos_keys_all3):
            break
        positive_keys3.append(pos_keys_all3[len(positive_keys3)])
    for _ in range(num_negative3):
        if len(negative_keys3) == len(neg_keys_all3):
            break
        negative_keys3.append(neg_keys_all3[len(negative_keys3)])
    if len(positive_keys3) < num_positive3:
        needed = num_positive3 - len(positive_keys3)
        extra = [k for k in pos_keys_all3 if k not in positive_keys3]
        positive_keys3 += extra[:needed]
    if len(negative_keys3) < num_negative3:
        needed = num_negative3 - len(negative_keys3)
        extra = [k for k in neg_keys_all3 if k not in negative_keys3]
        negative_keys3 += extra[:needed]

    pos_iter3 = itertools.cycle(positive_keys3) if len(positive_keys3) < num_positive3 else iter(positive_keys3)
    neg_iter3 = itertools.cycle(negative_keys3) if len(negative_keys3) < num_negative3 else iter(negative_keys3)
    used_pos3 = set()
    used_neg3 = set()
    for i, p in enumerate(seq3):
        if p == "positive":
            d = next(pos_iter3)
            while d in used_pos3 and len(used_pos3) < len(positive_keys3):
                d = next(pos_iter3)
            used_pos3.add(d)
        else:
            d = next(neg_iter3)
            while d in used_neg3 and len(used_neg3) < len(negative_keys3):
                d = next(neg_iter3)
            used_neg3.add(d)
        word = _rand.choice(desc[p][d])
        phrase = f"the {subj3}" if i == 0 else "it"
        verb = "was" if i < 8 else "is"
        sents3.append(f"{marks9[i]}, {phrase} {verb} {word}.")
        descs3.append(p)
        ints3.append(d)

    code_key3 = f"entity[[{subj3}_{ints3[0]}]]->entity[[{subj3}_{ints3[1]}]]->entity[[{subj3}_{ints3[2]}]]->entity[[{subj3}_{ints3[3]}]]->entity[[{subj3}_{ints3[4]}]]->entity[[{subj3}_{ints3[5]}]]->entity[[{subj3}_{ints3[6]}]]->entity[[{subj3}_{ints3[7]}]]->entity[[{subj3}_{ints3[8]}]]"

    out.append({
        "sentences": sents3,
        "description": "Aggregate packet (9)",
        "descriptor": descs3,
        "intensity": ints3,
        "code_key": code_key3,
        "entity": subj3,
        "marks": marks9,
        "type": "aggregate_long"
    })

    return out

def calibration_gen(seed=None, used_words=None):
    rnd = random.Random(seed)
    pos_pool = []
    for d in [pos_nouns, pos_verbs, pos_desc]:
        for v in d.values():
            pos_pool.extend(v)
    neg_pool = []
    for d in [neg_nouns, neg_verbs, neg_desc]:
        for v in d.values():
            neg_pool.extend(v)
    if used_words:
        pos_pool = [w for w in pos_pool if w not in used_words]
        neg_pool = [w for w in neg_pool if w not in used_words]
    if not pos_pool:
        for d in [pos_nouns, pos_verbs, pos_desc]:
            for v in d.values():
                pos_pool.extend(v)
    if not neg_pool:
        for d in [neg_nouns, neg_verbs, neg_desc]:
            for v in d.values():
                neg_pool.extend(v)
    pos_word = rnd.choice(pos_pool)
    neg_word = rnd.choice(neg_pool)
    def get_type(word, dicts):
        for dname, d in dicts:
            for k, v in d.items():
                if word in v:
                    return dname
        return None
    pos_type = get_type(pos_word, [('noun', pos_nouns), ('verb', pos_verbs), ('desc', pos_desc)])
    neg_type = get_type(neg_word, [('noun', neg_nouns), ('verb', neg_verbs), ('desc', neg_desc)])
    return pos_word, neg_word, pos_type, neg_type

def survey_gen(seed=None):
    if seed is not None: 
        _rand.seed(seed)
    else: 
        seed = _rand.randint(0, 1000000)
        _rand.seed(seed)
    
    used_names = set()
    used_objects = set()
    
    compound_actions = generate_compound_action_sentences(used_names)
    compound_associations = generate_compound_association_sentences(used_names)
    compound_belongings = generate_compound_belonging_sentences(used_objects)
    
    out = compound_actions + compound_associations + compound_belongings
    packets = generate_aggregate_sentiment_sentences(used_objects)
    
    return {
        "seed": seed, 
        "items": out, 
        "packets": packets
    }

# json.dump(survey_gen(), open("survey_data.json", "w"), indent=4, ensure_ascii=False)