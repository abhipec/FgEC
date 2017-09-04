"""
Data format conversion helper.
"""

def remove_overlapping_ems(mentions):
    """
    Spotlight can generate overlapping EMs.
    Among the intersecting EMs, remove the smaller ones.
    """
    to_remove = set()
    new_mentions = []
    length = len(mentions)
    for i in range(length):
        start_r = mentions[i]['start']
        end_r = mentions[i]['end']
        for j in range(length):
            if i != j and j not in to_remove:
                start = mentions[j]['start']
                end = mentions[j]['end']
                if start_r >= start and end_r <= end:
                    to_remove.add(i)
    for i in range(length):
        if i not in to_remove:
            new_mentions.append(mentions[i])
    return new_mentions

def generate_bio_entity_tags(tokens, mentions):
    """
    Generate BIO/IOB2 tags for entity detection task.
    """
    bio_tags = ['O'] * len(tokens)
    for mention in mentions:
        start = mention['start']
        end = mention['end']
        bio_tags[start] = 'B-E'
        for i in range(start + 1, end):
            bio_tags[i] = 'I-E'
    return bio_tags
