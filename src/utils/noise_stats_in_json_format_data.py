"""
Calculate various statistics of noisy labels in json format data.
"""

import sys
import json
import numpy as np

def labels_status(label_list):
    """
    Check is labels is clean or not.
    """
    if not label_list:
        return 1
    leaf = max(label_list, key=lambda x: x.count('/'))
    clean = 1
    for label_name in label_list:
        if label_name not in leaf:
            clean = 0
    return clean

#pylint:disable=invalid-name
if __name__ == '__main__':
    all_labels = {}

    with open(sys.argv[1], 'r', encoding='utf-8') as file_p:
        for row in file_p:
            json_data = json.loads(row)
            for mention in json_data['mentions']:
                for label in mention['labels']:
                    if not label in all_labels:
                        all_labels[label] = 0
                    all_labels[label] += 1
    print("Number of labels:", len(all_labels))
    print("Number of labels with more than 1000 mentions: ",
          len([x for x in all_labels if all_labels[x] > 1000]))
    print("Number of labels with more than 500 mentions: ",
          len([x for x in all_labels if all_labels[x] > 500]))
    print("Number of labels with more than 100 mentions: ",
          len([x for x in all_labels if all_labels[x] > 100]))

    top_level_labels = set([x for x in all_labels if x.count('/') == 1])
    leaf_level_labels = set([x for x in all_labels if x.count('/') == 2])

    total = 0
    noise_top_level, noise_leaf_level = 0, 0
    noise_top_level_count, noise_leaf_level_count = [], []
    clean_leaf_labels, clean_labels = 0, 0
    with open(sys.argv[1], 'r', encoding='utf-8') as file_p:
        for row in file_p:
            json_data = json.loads(row)
            for mention in json_data['mentions']:
                labels = set(mention['labels'])
                if len(labels.intersection(top_level_labels)) > 1:
                    noise_top_level += 1
                    noise_top_level_count.append(len(labels.intersection(top_level_labels)))
                if (len(labels.intersection(top_level_labels)) == 1 and
                        len(labels.intersection(leaf_level_labels)) > 1):
                    noise_leaf_level += 1
                    noise_leaf_level_count.append(len(labels.intersection(leaf_level_labels)))
                if len(labels) == 2 and labels_status(labels):
                    clean_leaf_labels += 1
                if labels_status(labels):
                    clean_labels += 1
                total += 1
    print("Percentage of top level noisy labels:", (noise_top_level / total) * 100)
    print("Mean no. of top level labels for noisy labels:", np.mean(noise_top_level_count))
    print("Median no. of top level labels for noisy labels:", np.median(noise_top_level_count))

    print("Percentage of leaf level noisy labels:", (noise_leaf_level / total) * 100)
    print("Mean no. of leaf level labels for noisy labels:", np.mean(noise_leaf_level_count))
    print("Median no. of leaf level labels for noisy labels:", np.median(noise_leaf_level_count))

    print("Percentage of clean labels:", (clean_labels / total) * 100)
    print("Percentage of clean labels at leaf level:", (clean_leaf_labels / total) * 100)
