import numpy as np
from mindspore import Tensor
from mindspore.nn import Cell,CellList
from sponge.functions import get_integer

def str_to_tensor(string:str) -> Tensor:
    if isinstance(string,(list,tuple)):
        string = ' '.join(string)
    return Tensor(np.fromstring(string,dtype=np.int8))

def tensor_to_str(tensor:Tensor) -> str:
    tensor = Tensor(tensor).asnumpy()
    string = tensor.tostring().decode()
    string = string.split()
    if len(string) == 1:
        string = string[0]
    return string

def get_class_parameters(hyper_param:dict,prefix:str,num_class:int=1) -> dict:
    def _get_class_parameters(hyper_param:dict,prefix:str) -> dict:
        new_params = {}
        idx = len(prefix) + 1
        for name,param in hyper_param.items():
            if name.find(prefix) == 0 \
                    and (name == prefix or name[len(prefix)] == "." or (prefix and prefix[-1] == ".")):
                new_params[name[idx:]] = param
        if 'name' in new_params.keys():
            new_params['name'] = get_hyper_string(new_params,'name')
            if len(new_params) == 1:
                new_params = new_params['name']

        if new_params:
            return new_params
        else:
            return None

    if num_class == 1:
        return _get_class_parameters(hyper_param,prefix)
    else:
        param_list = []
        for i in range(num_class):
            param_list.append(_get_class_parameters(hyper_param,prefix+'.'+str(i)))
        return param_list


def get_hyper_parameter(hyper_param:dict,prefix:str):
    if prefix in hyper_param.keys():
        return Tensor(hyper_param[prefix])
    else:
        return None

def get_hyper_string(hyper_param:dict,prefix:str):
    if prefix in hyper_param.keys():
        string = hyper_param[prefix]
        if isinstance(string,str):
            return string
        else:
            return tensor_to_str(string)
    else:
        return None

def set_hyper_parameter(hyper_param:dict,prefix:str,param:None):
    if param is None:
        if prefix in hyper_param.keys():
            hyper_param.pop(prefix)
    else:
        if isinstance(param,str):
            hyper_param[prefix] = str_to_tensor(param)
        else:
            hyper_param[prefix] = param

def set_class_parameters(hyper_param:list,prefix:str,cell:Cell):
    def _set_class_parameters(hyper_param:dict,prefix:str,cell:Cell):
        if isinstance(cell,Cell):
            if 'hyper_param' in cell.__dict__.keys():
                for key,param in cell.hyper_param.items():
                    set_hyper_parameter(hyper_param,prefix+'.'+key,param)
            else:
                set_hyper_parameter(hyper_param,prefix+'.name',cell.__class__.__name__)
        elif isinstance(cell,str):
            set_hyper_parameter(hyper_param,prefix,cell)
        elif cell is not None:
            raise TypeError('The type of "cls" must be "Cell", "str" or list of them, but got "'+
                str(type(cell))+'".')

    if isinstance(cell,(CellList,list)):
        for i,c in enumerate(cell):
            _set_class_parameters(hyper_param,prefix+'.'+str(i),c)
    else:
        _set_class_parameters(hyper_param,prefix,cell)

def load_hyper_param_into_class(cls_dict:dict,hyper_param:dict,types:dict,prefix:str=''):
    if len(prefix) > 0:
        prefix = prefix + '.'
    for key,value_type in types.items():
        if value_type == 'str':
            cls_dict[key] = get_hyper_string(hyper_param,prefix+key)
        elif value_type == 'int':
            cls_dict[key] = get_integer(hyper_param[prefix+key])
        elif value_type == 'class':
            num_class = 1
            num_key = 'num_' + key
            if num_key in cls_dict.keys():
                num_class = get_integer(cls_dict[prefix+num_key])
                cls_dict[key] = num_class
            cls_dict[key] = get_class_parameters(hyper_param,prefix+key,num_class)
        else:
            cls_dict[key] = get_hyper_parameter(hyper_param,prefix+key)

def set_class_into_hyper_param(hyper_param:dict,types:dict,cls:Cell,prefix:str=''):
    if len(prefix) > 0:
        prefix = prefix + '.'
    for key,value_type in types.items():
        if value_type == 'Cell':
            if key in cls._cells.keys():
                if cls._cells[key] is not None:
                    set_class_parameters(hyper_param,prefix+key,cls._cells[key])
        else:
            if key in cls.__dict__.keys():
                set_hyper_parameter(hyper_param,prefix+key,cls.__dict__[key])
            elif key in cls._tensor_list.keys():
                set_hyper_parameter(hyper_param,prefix+key,cls._tensor_list[key])