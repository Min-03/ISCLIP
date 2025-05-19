import spacy
import re


def _refine_class_name(class_name):
    if class_name == "tvmonitor":
        return ["tv monitor", "monitor", "tvmonitor"]
    elif class_name == "pottedplant":
        return ["potted plant", "pottedplant", "plant"]
    elif class_name == "aeroplane":
        return ["aeroplane", "airplane"]
    elif class_name == "diningtable":
        return ["diningtable", "dining table"]
    elif class_name == "motorbike":
        return ["motorbike", "motor bike", "bike"]
    else:
        return [class_name]

def _is_no_info(text, class_name):
    class_names = _refine_class_name(class_name)

    for class_name in class_names:
        pattern = r'^(a|an|the)?\s*' + re.escape(class_name) + r'(es|s)?$'
        if re.fullmatch(pattern, text.strip()) is not None:
            return True
    return False

def extract_noun_phrase(text, nlp, class_name):
    """
    Extracts target noun if the target noun is different from class_name, 
    otherwise it extracts detailed noun information related to target noun. 
    """
    text = text.lower().strip()
    doc = nlp(text)

    target_noun = extract_target_noun(doc, class_name)
    if _is_no_info(target_noun, class_name):
        detailed_noun = extract_detailed_noun(doc, class_name)
        return detailed_noun
    else:
        return target_noun

def extract_target_noun(doc, class_name):
    """
    Extracts first-occurring noun phrases
    """
    allowed_pos = {"ADJ", "NOUN", "CCONJ", "DET"}
    result_tokens = []

    for token in doc:
        if token.pos_ in allowed_pos:
            result_tokens.append(token)
        else:
            break 

    if len(result_tokens) == 0:
        return class_name

    res = " ".join([t.text for t in result_tokens]).strip()
    if result_tokens[-1].pos_ == "NOUN":
        return res
    else:
        return res + " " + class_name

def extract_detailed_noun(doc, class_name):
    """
    Extracts detailed noun that comes after target noun
    Ex. the car is a silver sedan --> silver sedan
    """
    is_index = next((i for i, token in enumerate(doc) if token.text == "is"), None)
    if is_index is None:
        return class_name

    after_is = doc[is_index + 1:]

    allowed_pos = {"ADJ", "NOUN", "CCONJ", "DET"}
    result_tokens = []

    for token in after_is:
        if token.pos_ in allowed_pos:
            result_tokens.append(token)
        else:
            break 

    if len(result_tokens) == 0:
        return class_name

    res = " ".join([t.text for t in result_tokens]).strip()
    if result_tokens[-1].text in _refine_class_name(class_name):
        return res
    else:
        return res + " " + class_name


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    print(extract_noun_phrase(" a blue bicycle is leaning against a white wall", nlp, "bicycle"))
    print(extract_noun_phrase("the monitor is a gray and white color", nlp, "tvmonitor"))