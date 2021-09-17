from Levenshtein import distance

def calculate_string_similarity(str1, str2):
    """Calculates the string similarity between str1 and str2.
    """
    str1 = str1 or ""
    str2 = str2 or ""
    str1, str2 = str1.lower(), str2.lower()
    if str1 == "" and str2 == "":
        return 1
    if str1 == "" or str2 == "":
        return 0
    return 1 - distance(str1, str2) / max(len(str1), len(str2))


def calculate_numeric_similarity(prop1, prop2, max_value):
    """Calculates the numeric similarity between prop1 and prop2.
    """

    return (
        prop1 and
        prop2 and
        max_value and
        (1.0 - abs(prop1 - prop2 - 0.0 ) / max_value) or 0.0
    )

def calculate_absolute_similarity(prop1, prop2):
    """Checks if prop1 and prop2 are the same.
    """

    if prop1 == prop2:
        return 1
    else:
        return 0

SIMILARITY_WEIGHTS = {
    "tag": 0.3,
    "name": 0.1,
    "class": 0.3,
    "text": 0.1,
    "x": 0.2,
    "y": 0.2,
    "attrs": 0.1,
    # "semantic": 0.4,
    # "extra_text": 0.58803,
}

def calculate_elements_similarity(e1,e2,driver,window_size,weights=SIMILARITY_WEIGHTS):
    sim_tag = calculate_absolute_similarity(e1.tag,e2.tag)

    sim_name = calculate_string_similarity(e1.name,e2.name)
    sim_class = calculate_absolute_similarity(e1.css_class,e2.css_class)
    sim_text = calculate_string_similarity(e1.text,e2.text)

    x1,y1 = e1.get_location(driver)
    x2,y2 = e2.get_location(driver)
    sim_x = calculate_numeric_similarity(x1,x2,window_size['width'])
    sim_y = calculate_numeric_similarity(y1,y2,window_size['height'])
    
    sim_attrs = 0
    common_keys = e1.attrs.keys() & e2.attrs.keys()
    for name in common_keys:
        attr1 = e1.attrs[name]
        attr2 = e2.attrs[name]
        sim_attrs += calculate_string_similarity(attr1,attr2) / len(common_keys)

    similarity_score = (
        (weights['tag'] * sim_tag +
        weights['name'] * sim_name +
        weights['class'] * sim_class +
        weights['text'] * sim_text +
        weights['x'] * sim_x +
        weights['y'] * sim_y +
        weights['attrs'] * sim_attrs) /
        sum(weights.values())
    )
    return similarity_score