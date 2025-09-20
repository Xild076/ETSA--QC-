import random, nltk
import itertools
from afinn import Afinn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
import ssl
import os
from pathlib import Path

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

current_file = Path(__file__)
project_root = current_file.parent.parent.parent 
lexicon_path = project_root / "src" / "sentiment" / "lexicons" / "optimized_lexicon.json"

fallback_paths = [
    "src/sentiment/lexicons/optimized_lexicon.json",
    "../sentiment/lexicons/optimized_lexicon.json", 
    "../../sentiment/lexicons/optimized_lexicon.json"
]

lexicon_file = None
if lexicon_path.exists():
    lexicon_file = str(lexicon_path)
else:
    for fallback in fallback_paths:
        if os.path.exists(fallback):
            lexicon_file = fallback
            break

if lexicon_file is None:
    raise FileNotFoundError("Could not find optimized_lexicon.json file")

loaded_lexicons = json.load(open(lexicon_file, "r"))

def extract_word_list(lexicon, key):
    lexicon_key = lexicon.get(key, {})
    return {
        valence: [entry["text"] for entry in entries if isinstance(entry, dict) and "text" in entry]
        for valence, entries in lexicon_key.items()
        if isinstance(entries, list)
    }

pos_nouns = extract_word_list(loaded_lexicons, "pos_nouns")
neg_nouns = extract_word_list(loaded_lexicons, "neg_nouns")
pos_verbs = extract_word_list(loaded_lexicons, "pos_verbs")
neg_verbs = extract_word_list(loaded_lexicons, "neg_verbs")
pos_desc = extract_word_list(loaded_lexicons, "pos_desc")
neg_desc = extract_word_list(loaded_lexicons, "neg_desc")

neutral_actions_together = ['ate', 'talked', 'worked', 'walked']


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
**Purpose:** To check x-x-y relationships.
**Text or Structure:**
"{+ actor} {+ action} {- victim}"
**Purpose:** To check y-x-x relationships
**Text or Structure:**
"{+ actor} {- action} {- victim}"
**Purpose:** To check x-x-x relationships
**Text or Structure:**
"{- actor} {- action} {- victim}"
**Purpose:** To check negativity biases, compared to x-y-x.
**Text or Structure:**
"{+ actor} {- action} {+ victim}"
"""

def _get_unique_words(word_pools, used_words_in_survey):
    selected_words = []
    local_used_words = set()

    for pool in word_pools:
        available_words = [w for w in pool if w not in used_words_in_survey and w not in local_used_words]
        if not available_words:
            available_words = [w for w in pool if w not in local_used_words]
        if not available_words:
            available_words = pool
        
        word = _rand.choice(available_words)
        selected_words.append(word)
        local_used_words.add(word)
    
    return selected_words

def generate_compound_action_sentences(used_names, used_words_in_survey):
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

        word1_pool = nouns[p[0]][i1]
        verb_pool = verbs[p[1]][i2]
        word3_pool = nouns[p[2]][i3]

        word1, verb, word3 = _get_unique_words([word1_pool, verb_pool, word3_pool], used_words_in_survey)
        
        used_words_in_survey.add(word1)
        used_words_in_survey.add(verb)
        used_words_in_survey.add(word3)

        sentence = f"{a}, {word1}, {verb} {v}, {word3}."
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
**Purpose:** To check intra-sentence opposite direction association
**Text or Structure:**
"{+ entity} and {- entity} {+ action} together."
**Purpose:** To check inter-sentence association and negativity bias. (Has some relation to aggregate sentiments, maybe there is an overlap). When doing calculations, maybe do aggregate sentiment first to get a formula for that, then cross-apply with this one to check for no temporal biases.
**Text or Structure:**
"{- entity} did {- action}. {+ entity} did it with them."
"""

def generate_compound_association_sentences(used_names, used_words_in_survey):
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

    word1_pool = nouns[x][i1]
    word2_pool = nouns[y][i2]
    word1, word2 = _get_unique_words([word1_pool, word2_pool], used_words_in_survey)
    used_words_in_survey.add(word1)
    used_words_in_survey.add(word2)

    sentence = f"{n1}, {word1}, and {n2}, {word2}, often {action} together."
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
    j2 = _rand.choice(list(nouns[x].keys()))

    word3_pool = nouns[x][j1]
    word4_pool = nouns[x][j2]
    word3, word4 = _get_unique_words([word3_pool, word4_pool], used_words_in_survey)
    used_words_in_survey.add(word3)
    used_words_in_survey.add(word4)

    sentence1 = f"{n3}, {word3}, often hung out with {n4}, {word4}."
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

    word5_pool = nouns[y][k1]
    word6_pool = nouns[y][k2]
    word5, word6 = _get_unique_words([word5_pool, word6_pool], used_words_in_survey)
    used_words_in_survey.add(word5)
    used_words_in_survey.add(word6)

    sentence2 = f"{n5}, {word5}, and {n6}, {word6}, {action2} together every weekend."
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

    word7_pool = nouns[y][l1]
    word8_pool = nouns[x][l2]
    word7, word8 = _get_unique_words([word7_pool, word8_pool], used_words_in_survey)
    used_words_in_survey.add(word7)
    used_words_in_survey.add(word8)

    sentence3 = f"{n7}, {word7}, partnered with {n8}, {word8} all the time."
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
**Purpose:** To check intra-sentence association
**Text or Structure:**
"{+ parent entity}'s {- child entity} was {- description}."
**Purpose:** To check inter-sentence association and negativity bias. (Has some relation to aggregate sentiments, maybe there is an overlap). When doing calculations, maybe do aggregate sentiment first to get a formula for that, then cross-apply with this one to check for no temporal biases.
**Text or Structure:**
"{- parent entity} was {- descriptor}. {+ child entity} was {+ description}."
"""

def generate_compound_belonging_sentences(used_objects, used_words_in_survey):
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
    
    word1_pool = desc[x][i1]
    word2_pool = desc[y][i2]
    word1, word2 = _get_unique_words([word1_pool, word2_pool], used_words_in_survey)
    used_words_in_survey.add(word1)
    used_words_in_survey.add(word2)
    
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
    
    word3_pool = desc[y][j1]
    word4_pool = desc[y][j2]
    word3, word4 = _get_unique_words([word3_pool, word4_pool], used_words_in_survey)
    used_words_in_survey.add(word3)
    used_words_in_survey.add(word4)

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

    word5_pool = desc[y][k1]
    word6_pool = desc[x][k2]
    word5, word6 = _get_unique_words([word5_pool, word6_pool], used_words_in_survey)
    used_words_in_survey.add(word5)
    used_words_in_survey.add(word6)

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
    l1 = _rand.choice(list(desc[x].keys()))
    l2 = _rand.choice(list(desc[x].keys()))

    word7_pool = desc[x][l1]
    word8_pool = desc[x][l2]
    word7, word8 = _get_unique_words([word7_pool, word8_pool], used_words_in_survey)
    used_words_in_survey.add(word7)
    used_words_in_survey.add(word8)

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
*3 questions*
**Purpose:** To check shorter aggregate sentiment shifts.
**Text or Structure:**
1. "xxx was good."
2. "xxx was bad."
3. "xxx was good."
*5 questions*
**Purpose:** To check longer aggregate sentiment shifts and possible negative biases.
**Text or Structure:**
1. "xxx was bad."
2. "xxx was good."
3. "xxx was good."
4. "xxx was bad."
5. "xxx was bad."
"""

def generate_aggregate_sentiment_sentences(used_objects, used_words_in_survey):
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
        
        word_pool = desc[p][d]
        word = _get_unique_words([word_pool], used_words_in_survey)[0]
        used_words_in_survey.add(word)

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
        
        word_pool = desc[p][d]
        word = _get_unique_words([word_pool], used_words_in_survey)[0]
        used_words_in_survey.add(word)

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
        
        word_pool = desc[p][d]
        word = _get_unique_words([word_pool], used_words_in_survey)[0]
        used_words_in_survey.add(word)

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

def calibration_gen(used_objects, used_words_in_survey):
    pos_pool = {k: nouns['positive'][k] + verbs['positive'][k] + desc['positive'][k] for k in nouns['positive']}
    neg_pool = {k: nouns['negative'][k] + verbs['negative'][k] + desc['negative'][k] for k in nouns['negative']}
    intensity_pos = _rand.choice(list(pos_pool.keys()))
    intensity_neg = _rand.choice(list(neg_pool.keys()))
    pos_word_pool = pos_pool[intensity_pos]
    neg_word_pool = neg_pool[intensity_neg]
    pos_word = _get_unique_words([pos_word_pool], used_words_in_survey)[0]
    neg_word = _get_unique_words([neg_word_pool], used_words_in_survey)[0]
    return {
        "positive": {
            "word": pos_word,
            "intensity": intensity_pos,
            "code_key": f"calibration_pos[[{pos_word}_{intensity_pos}]]"
        },
        "negative": {
            "word": neg_word,
            "intensity": intensity_neg,
            "code_key": f"calibration_neg[[{neg_word}_{intensity_neg}]]"
        }
    }

def survey_gen(seed=None):
    if seed is not None: 
        _rand.seed(seed)
    else: 
        seed = _rand.randint(0, 1000000)
        _rand.seed(seed)
    
    used_names = set()
    used_objects = set()
    used_words_in_survey = set()

    calibration = calibration_gen(used_objects, used_words_in_survey)
    
    compound_actions = generate_compound_action_sentences(used_names, used_words_in_survey)
    compound_associations = generate_compound_association_sentences(used_names, used_words_in_survey)
    compound_belongings = generate_compound_belonging_sentences(used_objects, used_words_in_survey)
    
    out = compound_actions + compound_associations + compound_belongings
    packets = generate_aggregate_sentiment_sentences(used_objects, used_words_in_survey)
    
    return {
        "seed": seed, 
        "calibration": calibration,
        "items": out, 
        "packets": packets
    }

