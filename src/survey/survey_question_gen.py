import random, nltk
from afinn import Afinn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json

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
    "very": "a brave, kind, and brilliant person",
    "medium": "a great person",
    "somewhat": "a thoughtful person",
    "slightly": "a somewhat considerate person"
}

neg_nouns = {
    "very": "an utterly vile and disgusting person",
    "medium": "an unpleasant person",
    "somewhat": "a problematic person",
    "slightly": "a nice but annoying person"
}

pos_verbs = {
    "very": "heroically and bravely rescued",
    "medium": "kindly saved",
    "somewhat": "encouraged",
    "slightly": "lightly comforted"
}

neg_verbs = {
    "very": "brutally murdered",
    "medium": "abused",
    "somewhat": "frustrated",
    "slightly": "mildly annoyed"
}

pos_desc = {
    "very": ["superb", "outstanding", "breathtaking"],
    "medium": ["great", "excellent", "terrific"],
    "somewhat": ["fine", "fair", "somewhat nice"],
    "slightly": ["okay", "alright", "acceptable"]
}

neg_desc = {
    "very": ["atrocious", "horrendous", "appalling"],
    "medium": ["bad", "awful", "defective"],
    "somewhat": ["poor", "broken", "annoying"],
    "slightly": ["unimpressive", "problematic", "not great"]
}

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

objs = ["phone", "laptop", "game", "car"]

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
        
        sentence = f"{a}, {nouns[p[0]][i1]}, {verbs[p[1]][i2]} {v}, {nouns[p[2]][i3]}."
        code_key = f"actor[[{a}_{i1}]]->verb[[{verbs[p[1]][i2]}_{i2}]]->target[[{v}_{i3}]]"
        
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
**Purpose:** To check intra-sentence association
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
    if len(available_names) < 3:
        available_names = names
    n1, n2, n3 = _rand.sample(available_names, 3)
    used_names.update([n1, n2, n3])
    
    i1 = _rand.choice(list(nouns[x].keys()))
    i2 = _rand.choice(list(nouns[y].keys()))
    i3 = _rand.choice(list(verbs[x].keys()))
    
    sentence = f"{n1}, {nouns[x][i1]}, and {n2}, {nouns[y][i2]}, together {verbs[x][i3]} {n3}."
    code_key = f"actor[[{n1}_{i1}]]+actor[[{n2}_{i2}]]->verb[[{verbs[x][i3]}_{i3}]]->target[[{n3}_N]]"
    
    out.append({
        "sentences": [sentence],
        "description": "Intra-sentence association",
        "descriptor": [x, y, x, "neutral"],
        "intensity": [i1, i2, i3, "neutral"],
        "entities": [n1, n2, n3],
        "code_key": code_key,
        "marks": [],
        "type": "compound_association"
    })
    
    available_names = [n for n in names if n not in used_names]
    if len(available_names) < 3:
        available_names = [n for n in names if n not in [n1, n2, n3]]
    n4, n5, n6 = _rand.sample(available_names, 3)
    used_names.update([n4, n5, n6])
    
    j1 = _rand.choice(list(nouns[y].keys()))
    j2 = _rand.choice(list(verbs[x].keys()))
    j3 = _rand.choice(list(nouns[y].keys()))
    
    sentence1 = f"{n4}, {nouns[y][j1]}, {verbs[x][j2]} {n5}."
    sentence2 = f"{n6}, {nouns[y][j3]}, did it with {n4}."
    code_key = f"actor[[{n4}_{j1}]]+actor[[{n6}_{j3}]]->verb[[{verbs[x][j2]}_{j2}]]->target[[{n5}_N]]"
    
    out.append({
        "sentences": [sentence1, sentence2],
        "description": "Inter-sentence association",
        "descriptor": [y, x, y, "neutral"],
        "intensity": [j1, j2, j3, "neutral"],
        "entities": [n4, n5, n6],
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
    
    sentence = f"The {parent}'s {child} was {word2}, though the {parent} itself was {word1}."
    code_key = f"parent[[{parent}_{i1}]]->child[[{child}_{i2}]]"

    out.append({
        "sentences": [sentence],
        "description": "Intra-sentence belonging",
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
    j2 = _rand.choice(list(desc[x].keys()))
    word3 = _rand.choice(desc[y][j1])
    word4 = _rand.choice(desc[x][j2])
    
    sentence1 = f"Once, the {parent2} was {word3}."
    sentence2 = f"Now, its {child2} is {word4}."
    code_key = f"parent[[{parent2}_{j1}]]->child[[{child2}_{j2}]]"

    out.append({
        "sentences": [sentence1, sentence2],
        "description": "Inter-sentence belonging",
        "descriptor": [y, x],
        "intensity": [j1, j2],
        "code_key": code_key,
        "entities": [parent2, child2],
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

    for i, p in enumerate(seq):
        d = _rand.choice(list(desc[p].keys()))
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
    
    y2, x2 = _rand.sample(["positive", "negative"], 2)
    seq2 = [y2, x2, x2, y2, y2]
    temporal_marker_sets_5 = [
        ["First", "Second", "Third", "Fourth", "Finally"],
        ["Once", "Soon after", "For a while", "Recently", "Now"],
        ["Initially", "Subsequently", "After some time", "More recently", "Currently"]
    ]

    marks5 = _rand.choice(temporal_marker_sets_5)
    sents2, descs2, ints2 = [], [], []
    
    for i, p in enumerate(seq2):
        d = _rand.choice(list(desc[p].keys()))
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
        "type": "aggregate_long"
    })
    
    return out

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