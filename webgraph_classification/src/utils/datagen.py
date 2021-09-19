import json
import numpy as np
import numpy.random as random
import string

class DataGenerator():
    
    def __init__(self,datagen_path):
        with open(datagen_path) as fp:
            self.fields = json.load(fp)['fields']
        self.bisect_idx = 0


    def infer_random_like(self,value,range_str=None):
        to_type = type(value)
        rng = parse_range(value,range_str)
        if isinstance(value,int):
            start,stop,step = rng 
            interval = np.arange(start,stop,step)
        elif isinstance(value,float):
            start,stop,step = rng 
            interval = np.arange(start,stop,step)
        elif isinstance(value,str):
            interval = rng
        else:
            interval = ['0']
        
        return to_type(random.choice(interval))

    def infer_out_of_range(self,value,range_str=None):
        to_type = type(value)
        rng = parse_range(value,range_str)
        if isinstance(value,int):
            start,stop,_ = rng 
            interval = np.arange(start-100,start).tolist() + np.arange(stop,stop+100).tolist()
        elif isinstance(value,float):
            start,stop,_ = rng 
            interval = np.arange(start-100,start).tolist() + np.arange(stop,stop+100).tolist()
        elif isinstance(value,str):
            interval = [''.join(random.choice([c for c in (string.ascii_letters + string.digits)], size=10))]
        else:
            interval = ['0']
        
        return to_type(random.choice(interval))
    
    def infer_bisected(self,value,range_str=None):
        to_type = type(value)
        rng = parse_range(value,range_str)
        if isinstance(value,(int,float)):
            start,stop,_ = rng 
            interval = get_bisected_range(start,stop)
        else:
            return None
        
        if self.bisect_idx >= len(interval):
            self.bisect_idx = 0
        result = to_type(interval[self.bisect_idx])
        self.bisect_idx +=1
        return result

    def infer(self, driver, element):
        """
        WebDriver element
        """
        url = driver.get_current_url()
        data = self.filter_url(url)
        data = self.get_field_like_element(driver,data,element)

        if data:
            return data['value']
        return data

    def get_field_like_element(self,driver,data,element):
        result = [d for d in data if element == driver.get_element('xpath',d['xpath'])]
        if len(result) == 1:
            return result[0]
        return None

    def filter_url(self,url):
        return [field for field in self.fields if field['url'] == url]

def gen_uniform(start,stop):
    return random.randint(start,stop)


def get_bisected_range(start,stop, n=9):
    result = []
    mid = (start+stop) // 2
    result.append(start)
    result.append(stop)
    result.append(mid)
    mid_l,mid_r = mid,mid
    for _ in range((n-2)//2):
        mid_l = (start+mid_l) // 2
        mid_r = (mid_r+stop) // 2
        result.append(mid_l)
        result.append(mid_r)
    return sorted(result)

def parse_range(initial,range_str):
    
    if isinstance(initial,(int,float)):
        if range_str is None:
            return 0,100,None

        to_type = type(initial)
        splitted = range_str.split(',')
        assert len(splitted) == 2,f'Cannot parse range {range_str}'
        first,last = splitted

        if first[0] == '[':
            first = to_type(first[1:])
        elif first[0] == '(':
            first = to_type(first[1:]) + 1
        else:
            raise ValueError(f'Cannot parse range {range_str}')
        step = None
        last = last.split(':')
        if len(last) == 2:
            step = to_type(last[1][:-1])
            last = last[0] + last[1][-1]
        else:
            last = last[0]
        if last[-1] == ']':
            last = to_type(last[:-1]) + 1
        elif last[-1] == ')':
            last = to_type(last[:-1])
        else:
            raise ValueError(f'Cannot parse range {range_str}')
        
        return first, last, step
    elif isinstance(initial,str):
        if range_str is None:
            return [''.join(random.choice([c for c in (string.ascii_letters + string.digits )], size=10))]
        strings = range_str[1:-1].split(',')
        return strings
    else:
        raise ValueError(f'Cannot parse range {range_str}')
