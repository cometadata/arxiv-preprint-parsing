import re
from prompt import Affiliations
import json
from rapidfuzz import fuzz

LARGEST_LIST_RE = re.compile(r"<\/think>[^\[]*(\[.*\])", re.DOTALL)

def parse_completion(completion):
    # find the answer in the completion
    try:
        match = LARGEST_LIST_RE.search(completion).group(1)
    except Exception:
        return None

    # check to see if it parses as JSON
    try:
        match = json.loads(match)
    except json.JSONDecodeError:
        return None

    # check to see if it matches the schema
    try:
        Affiliations.model_validate(match)
    except Exception:
        return None

    return match


def format_reward(completions, **kwargs):
    rewards = []

    for completion in completions:
        if parse_completion(completion[0]['content']) is None:
            rewards.append(0)
        else:
            rewards.append(1)

    return rewards


def match_fuzzy(source, target, threshold = 90):
    """
    Fuzzy match two lists of authors and return a mapping from source index to target index.
    """
    mapping = {}
    for i, s in enumerate(source):
        best_match = None
        best_score = 0
        for j, t in enumerate(target):
            score = fuzz.ratio(s, t)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = j

        if best_match is not None:
            mapping[i] = best_match

    return mapping


def answer_reward(completions, answer, **kwargs):
    rewards = []

    answers = answer

    for completion, answer in zip(completions, answers):
        completion = completion[0]['content'] # message dictionary -> text of message
        print(completion)
        prediction = parse_completion(completion)
        if prediction is None:
            rewards.append(0)
            continue

        r = 0.0

        # intersection over union for authors
        found_authors = [x['name'] for x in prediction]
        gt_authors = [x['name'] for x in answer]
        mapping = match_fuzzy(gt_authors, found_authors)

        unmatched_source = set(range(len(gt_authors))) - set(mapping.keys())
        unmatched_target = set(range(len(found_authors))) - set(mapping.values())
        iou_authors = len(mapping) / (len(unmatched_source) + len(unmatched_target) + len(mapping))
        r += iou_authors

        # for each of the matched authors, match affiliations and increase reward by iou
        for source_idx, target_idx in mapping.items():
            source_affiliations = answer[source_idx]['affiliations']
            target_affiliations = prediction[target_idx]['affiliations']
            if not source_affiliations or not target_affiliations:
                continue

            # fuzzy match
            affiliation_mapping = match_fuzzy(source_affiliations, target_affiliations)
            unmatched_source = set(range(len(source_affiliations))) - set(affiliation_mapping.keys())
            unmatched_target = set(range(len(target_affiliations))) - set(affiliation_mapping.values())
            iou_affiliations = len(affiliation_mapping) / (len(unmatched_source) + len(unmatched_target) + len(affiliation_mapping))
            r += iou_affiliations

        rewards.append(r)

    return rewards
