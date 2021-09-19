import logging
from bs4 import BeautifulSoup
from src.utils import soup2xpath

logger = logging.getLogger(__name__)

def parse(html):
    soup = BeautifulSoup(html, 'lxml')
    if len(soup.select(f"[class*=Mui]")) != 0:
        parser = MaterialUIParser()
    else:
        parser = SimpleUIParser()
        
    return parser.parse(soup)

class BaseUIParser():
    def parse(self,soup_obj):
        raise NotImplementedError()

    def _create_dict(self,clickables,enterables):
        return {**{soup2xpath(e):'enterable' for e in enterables}, 
                **{soup2xpath(e):'clickable' for e in clickables}}

class SimpleUIParser(BaseUIParser):
    def __init__(self):
        self.clickable_elements ={'tag':['button','a']}
        self.enterable_elements = {'tag':['input','textarea']}
        logger.info('Using simple UI parser!')

    def parse(self,soup_obj):
        clickable = []
        for tag in self.clickable_elements['tag']:
            clickable.extend(soup_obj.select(f'{tag}'))
        
        enterable = []
        for tag in self.enterable_elements['tag']:
            enterable.extend(soup_obj.select(f'{tag}'))
        
        return self._create_dict(clickable,enterable)
     
class MaterialUIParser(BaseUIParser):
    def __init__(self):
        self.clickable_elements = {'class':['MuiButtonBase','MuiListItem']}
        self.enterable_elements = {'class':['MuiInputBase-input']}
        logger.info('Using MUI parser!')

    def parse(self,soup_obj):

        clickable = []
        for cls in self.clickable_elements['class']:
            elems = soup_obj.select(f'[class*={cls}]')
            clickable.extend(self._filter_elements(elems))
        
        enterable = []
        for cls in self.enterable_elements['class']:
            elems = soup_obj.select(f'[class*={cls}]')
            enterable.extend(self._filter_elements(elems))
        
        return self._create_dict(clickable,enterable)
 
    def _filter_elements(self,elems):
        new_elems = []
        if elems:
            prev = elems[0]
            for e in elems[1:]:
                if prev not in e.parents:
                    new_elems.append(prev)
                    prev = e
            new_elems.append(prev)
        return new_elems