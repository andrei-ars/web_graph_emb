import lxml
from lxml import etree
import formasaurus


def classify_html(html):
    html = _preprocess(html)
    extract = formasaurus.extract_forms(html)
    return [ {'form':d['form'], 'fields':{v:k for (k,v) in d['fields'].items()}} for (_,d) in extract]

def _preprocess(html):
    parsed = lxml.html.fromstring(html)
    for i,node in enumerate(parsed.xpath('//button')):
        new_node = etree.Element("input")
        new_node.set('name',f'button[{i}]')
        for k,v in node.items():
            new_node.set(k, v)
        node.getparent().replace(node, new_node)

    for i, node in enumerate(parsed.xpath('//input')):
        if 'name' not in node.attrib:
            if 'type' in node.attrib:
                node.set('name',node.get('type'))
            else:
                node.set('name',f'input{i}')
    return parsed