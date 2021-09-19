from src.utils import soup2xpath

from feature_extraction.extractor import FeatureExtractor

class PageElement():
    COPYABLE_ATTRIBUTES = ['available']

    def __init__(self,id,tag,attrs,interaction_type,text=None,xpath=None):
        self.id = id

        self.tag = tag
        self.dom_id = attrs.get('id',None)
        self.name = attrs.get('name',None)
        self.css_class = attrs.get('class',None)
        
        self.text = None
        if text:
            self.text = text
        elif attrs.get('value',None):
            self.text = attrs['value']
        elif attrs.get('placeholder',None):
            self.text = attrs['placeholder']
        
        self.attrs =  {k: v for k, v in attrs.items() if k not in ['id','name','class','value','placeholder']}

        self.xpath = xpath

        self._x,self._y = -1,-1


        self.options = None
        if isinstance(interaction_type,tuple):
            interaction_type,elements = interaction_type
            self.options = elements
            
        self.interaction_type = interaction_type
        

    @classmethod
    def from_bs4_object(cls,id,bs4_object,parsed_types):
        xpath = soup2xpath(bs4_object)
        return cls(id,str(bs4_object.name),bs4_object.attrs,parsed_types.get(xpath,'none'),bs4_object.text,xpath)

    def get_location(self,driver):
        if self._x == -1 and self._y == -1:
            self._x,self._y = driver.get_location(self)
        return self._x, self._y

    def is_available(self,driver):
        if self.interaction_type == 'none':
            return False
        return driver.element_clickable(self)

    def copy_attributes(self,element):
        for attr in PageElement.COPYABLE_ATTRIBUTES:
            value = element.__getattribute__(attr)
            self.__setattr__(attr,value)  

    def get_features(self):
        vec = FeatureExtractor.extract(self)
        return vec

    def __str__(self):
        return f'<{self.tag} ' + \
                (f'id=\'{self.dom_id}\' ' if self.dom_id else '') + \
                (f'name=\'{self.name}\' ' if self.name else '') + \
                (f'text=\'{self.text}\'' if self.text else '') + '>'
                # (f'xpath=\'{self.xpath}\' ' if self.xpath else '') + 

    def __hash__(self):
        return hash((self.id, self.tag,self.dom_id, self.name,self.text,self.xpath))

    def __eq__(self, other):
        return (self.id, self.tag, self.dom_id, self.name,self.text,self.xpath) == \
               (other.id, other.tag,self.dom_id, other.name,other.text,other.xpath)    

    def _pickle_value(self,k,v):
        return v

    def __getstate__(self):
        return {k:self._pickle_value(k,v) for (k, v) in self.__dict__.items() }