# ============================================================================
# Copyright 2021 The AIMM team at Shenzhen Bay Laboratory & Peking University
#
# People: Yi Isaac Yang, Jun Zhang, Diqing Chen, Yaqiang Zhou, Huiyang Zhang,
#         Yupeng Huang, Yijie Xia, Yao-Kun Lei, Lijiang Yang, Yi Qin Gao
#
# This code is a part of Cybertron-Code package.
#
# The Cybertron-Code is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""utils"""
import math

# origin constant
AVOGADRO_NUMBER = 6.02214076e23           # N_A
BOLTZMANN_CONSTANT = 1.380649e-23         # kB
GAS_CONSTANT = 8.31446261815324           # R unit=1/mol
ELEMENTARY_CHARGE = 1.602176634e-19       # e unit=C
VACCUM_PERMITTIVITY = 8.854187812813e-12  # \epsilon_0
COULOMB_CONSTANT = 8.9875517923e9         # k = 1/(4*pi*\epsilon_0) unit=N*m^2/C^2

_LENGTH_UNITS = (
    'nm',
    'um',
    'a',
    'angstrom',
    'bohr',
    'user',
    'none',
)

_LENGTH_REF = {
    'nm': 1.0,
    'um': 1e3,
    'a': 0.1,
    'angstrom': 0.1,
    'bohr': 0.052917721067,
    'user': None,
    'none': None,
}

_LENGTH_NAME = {
    'nm': 'nm',
    'um': 'um',
    'a': 'Angstrom',
    'bohr': 'Bohr',
    'user': 'User_Length',
    'none': "None"
}

_ENERGY_UNITS = (
    'kj/mol',
    'j/mol',
    'kcal/mol',
    'cal/mol',
    'ha',
    'ev',
    'kbt0',
    'kbt300',
    'user',
    'none',
)

_ENERGY_REF = {
    'kj/mol': 1.0,
    'j/mol': 1e-3,
    'kcal/mol': 4.184,
    'cal/mol': 4.184e-3,
    'ha': 2625.5002,
    'ev': 96.48530749925793,
    'kbt0': 2.271095464,
    'kbt300': 2.494338785,
    'user': None,
    'none': None,
}

_ENERGY_NAME = {
    'kj/mol': 'kJ/mol',
    'j/mol': 'J/mol',
    'kcal/mol': 'kcal/mol',
    'cal/mol': 'cal/mol',
    'ha': 'Hartree',
    'ev': 'eV',
    'kbt0': 'kBT(273.15K)',
    'kbt300': 'kBT(300K)',
    'user': 'User_Energy',
    'none': 'None',
}

# Boltzmann constant for simulation
_BOLTZMANN_DEFAULT_REF = 8.31446261815324e-3  # for kJ/mol
# Coulomb constant for simulation
# N_A*e^2/(4*pi*\epsilon_0)*1e9nm[1m]*1e-3kJ[1J] unit=e^2*kJ/mol*nm
_COULOMB_DEFAULT_REF = 138.93545764498226165718756672623

class Length:
    """Length"""
    def __init__(self,
        value,
        unit='nm',
    ):
        if isinstance(value,Length):
            self.__value = value.value
            self.__unit = value.unit
            self.__ref = value.ref
            self.__abs_size = value.abs_size
            self.__unit_name = value.unit_name
        elif isinstance(value,(float,int)):
            self.__value = float(value)
            if isinstance(unit,(str,Units)):
                self.__unit = get_length_unit(unit)
                self.__ref = get_length_ref(unit)
            elif isinstance(unit,(float,int)):
                self.__unit = 'user'
                self.__ref = float(unit)
            else:
                raise TypeError('Unsupported length unit type: ' + str(type(unit)))
            self.__abs_size = self.__value * self.__ref
            self.__unit_name = get_length_unit_name(self.__unit)
        else:
            raise TypeError('Unsupported length value type: ' + str(type(value)))

    def change_unit(self,unit):
        if isinstance(unit,(str,Units)):
            self.__unit = get_length_unit(unit)
            self.__ref = get_length_ref(unit)
        elif isinstance(unit,(float,int)):
            self.__unit = 'user'
            self.__ref = unit
        else:
            raise TypeError('Unsupported length unit type: ' + str(type(unit)))
        self.__value = length_convert('nm',unit) * self.__abs_size
        self.__unit_name = get_length_unit_name(self.__unit)
        return self

    @property
    def abs_size(self):
        return self.__abs_size

    @property
    def value(self):
        return self.__value

    @property
    def ref(self):
        return self.__ref

    @property
    def unit(self):
        return self.__unit

    @property
    def unit_name(self):
        return self.__unit_name

    def __call__(self,unit=None):
        return self.__value * length_convert(self.__unit,unit)

    def __str__(self):
        return str(self.__value) + ' ' + self.__unit_name

    def __lt__(self,other):
        if isinstance(other,Length):
            return self.__abs_size < other.abs_size
        else:
            return self.__value < other

    def __gt__(self,other):
        if isinstance(other,Length):
            return self.__abs_size > other.abs_size
        else:
            return self.__value > other

    def __eq__(self,other):
        if isinstance(other,Length):
            return self.__abs_size == other.abs_size
        else:
            return self.__value == other

    def __le__(self,other):
        if isinstance(other,Length):
            return self.__abs_size <= other.abs_size
        else:
            return self.__value <= other

    def __ge__(self,other):
        if isinstance(other,Length):
            return self.__abs_size >= other.abs_size
        else:
            return self.__value >= other

class Energy:
    """Energy"""
    def __init__(self,
        value,
        unit='kj/mol',
    ):
        if isinstance(value,Energy):
            self.__value = value.value
            self.__unit = value.unit
            self.__ref = value.ref
            self.__abs_size = value.abs_size
            self.__unit_name = value.unit_name
        elif isinstance(value,(float,int)):
            self.__value = float(value)
            if isinstance(unit,(str,Units)):
                self.__unit = get_energy_unit(unit)
                self.__ref = get_energy_ref(unit)
            elif isinstance(unit,(float,int)):
                self.__unit = 'user'
                self.__ref = float(unit)
            else:
                raise TypeError('Unsupported energy unit type: ' + str(type(unit)))
            self.__abs_size = self.__value * self.__ref
            self.__unit_name = get_energy_unit_name(self.__unit)
        else:
            raise TypeError('Unsupported energy value type: ' + str(type(value)))

    def change_unit(self,unit):
        if isinstance(unit,(str,Units)):
            self.__unit = get_energy_unit(unit)
            self.__ref = get_energy_ref(unit)
        elif isinstance(unit,(float,int)):
            self.__unit = 'user'
            self.__ref = unit
        else:
            raise TypeError('Unsupported energy unit type: ' + str(type(unit)))
        self.__value = energy_convert('kj/mol',unit) * self.__abs_size
        self.__unit_name = get_energy_unit_name(self.__unit)
        return self

    def __call__(self,unit=None):
        return self.__value * energy_convert(self.__unit,unit)

    def __str__(self):
        return str(self.__value) + ' ' + self.__unit_name

    def __lt__(self,other):
        if isinstance(other,Energy):
            return self.__abs_size < other.abs_size
        else:
            return self.__value < other

    def __gt__(self,other):
        if isinstance(other,Energy):
            return self.__abs_size > other.abs_size
        else:
            return self.__value > other

    def __eq__(self,other):
        if isinstance(other,Energy):
            return self.__abs_size == other.abs_size
        else:
            return self.__value == other

    def __le__(self,other):
        if isinstance(other,Energy):
            return self.__abs_size <= other.abs_size
        else:
            return self.__value <= other

    def __ge__(self,other):
        if isinstance(other,Energy):
            return self.__abs_size >= other.abs_size
        else:
            return self.__value >= other

    @property
    def abs_size(self):
        return self.__abs_size

    @property
    def value(self):
        return self.__value

    @property
    def ref(self):
        return self.__ref

    @property
    def unit(self):
        return self.__unit

    @property
    def unit_name(self):
        return self.__unit_name

def get_length_ref(unit):
    if unit is None:
        return None
    if isinstance(unit,str):
        if unit.lower() not in _LENGTH_REF.keys():
            raise KeyError('length unit "' + unit + '" is not recorded!')
        return _LENGTH_REF[unit.lower()]
    elif isinstance(unit,Units):
        return unit.length_ref
    elif isinstance(unit,Length):
        return unit.ref
    elif isinstance(unit,(float,int)):
        return unit
    else:
        raise TypeError('Unsupported length reference type: ' + str(type(unit)))

def get_length_unit(unit):
    if unit is None:
        return 'none'
    if isinstance(unit,str):
        if unit.lower() not in _LENGTH_UNITS:
            raise KeyError('length unit "' + unit + '" is not recorded!')
        return unit.lower()
    elif isinstance(unit,Units):
        return unit.length_unit
    elif isinstance(unit,Length):
        return unit.unit
    elif isinstance(unit,(float,int)):
        return 'user'
    else:
        raise TypeError('Unsupported length unit type: ' + str(type(unit)))

def get_length_unit_name(unit):
    if unit is None:
        return 'None'
    if isinstance(unit,str):
        if unit.lower() not in _LENGTH_NAME.keys():
            raise KeyError('length unit "' + unit + '" is not recorded!')
        return _LENGTH_NAME[unit.lower()]
    elif isinstance(unit,Units):
        return unit.length_unit_name
    elif isinstance(unit,Length):
        return unit.unit_name
    elif isinstance(unit,(float,int)):
        return 'User_Length'
    else:
        raise TypeError('Unsupported length unit name type: ' + str(type(unit)))

def get_energy_ref(unit):
    if unit is None:
        return None
    if isinstance(unit,str):
        if unit.lower() not in _ENERGY_REF.keys():
            raise KeyError('energy unit "' + unit + '" is not recorded!')
        return _ENERGY_REF[unit.lower()]
    elif isinstance(unit,Units):
        return unit.energy_ref
    elif isinstance(unit,Energy):
        return unit.ref
    elif isinstance(unit,(float,int)):
        return unit
    else:
        raise TypeError('Unsupported energy reference type: ' + str(type(unit)))

def get_energy_unit(unit):
    if unit is None:
        return 'none'
    if isinstance(unit,str):
        if unit.lower() not in _ENERGY_UNITS:
            raise KeyError('energy unit "' + unit + '" is not recorded!')
        return unit.lower()
    elif isinstance(unit,Units):
        return unit.energy_unit
    elif isinstance(unit,Energy):
        return unit.unit
    elif isinstance(unit,(float,int)):
        return 'user'
    else:
        raise TypeError('Unsupported energy unit type: ' + str(type(unit)))

def get_energy_unit_name(unit):
    if unit is None:
        return 'None'
    if isinstance(unit,str):
        if unit.lower() not in _ENERGY_NAME.keys():
            raise KeyError('energy unit "' + unit + '" is not recorded!')
        return _ENERGY_NAME[unit.lower()]
    elif isinstance(unit,Units):
        return unit.energy_unit_name
    elif isinstance(unit,Energy):
        return unit.unit_name
    elif isinstance(unit,(float,int)):
        return 'User_Energy'
    else:
        raise TypeError('Unsupported energy unit name type: ' + str(type(unit)))

def length_convert(unit_in, unit_out):
    length_in = get_length_ref(unit_in)
    length_out = get_length_ref(unit_out)
    if length_in is None or length_out is None:
        return 1
    else:
        return  length_in / length_out

def energy_convert(unit_in, unit_out):
    energy_in = get_energy_ref(unit_in)
    energy_out = get_energy_ref(unit_out)
    if energy_in is None or energy_out is None:
        return 1
    else:
        return  energy_in / energy_out

class Units:
    """Units"""
    def __init__(self,
        length_unit=None,
        energy_unit=None,
    ):

        self.__length_unit = get_length_unit(length_unit)
        self.__length_unit_name = get_length_unit_name(length_unit)
        self.__length_ref = get_length_ref(length_unit)
    
        self.__energy_unit = get_energy_unit(energy_unit)
        self.__energy_unit_name = get_energy_unit_name(energy_unit)
        self.__energy_ref = get_energy_ref(energy_unit)

        self.__boltzmann = _BOLTZMANN_DEFAULT_REF
        if self.__energy_ref is not None:
            self.__boltzmann /= self.__energy_ref
        self.__coulomb = _COULOMB_DEFAULT_REF
        if self.__length_ref is not None and self.__energy_ref is not None:
            self.__coulomb *= self.__energy_ref * self.__length_ref

    def set_length_unit(self, unit=None):
        if unit is not None:
            self.__length_unit = get_length_unit(unit)
            self.__length_unit_name = get_length_unit_name(unit)
            self.__length_ref = get_length_ref(unit)
            self.set_depends()
        return self

    def set_energy_unit(self, unit=None):
        if unit is not None:
            self.__energy_unit = get_energy_unit(unit)
            self.__energy_unit_name = get_energy_unit_name(unit)
            self.__energy_ref = get_energy_ref(unit)
            self.set_depends()
        return self

    def set_units(self,length_unit,energy_unit,units=None):
        if units is None:
            if length_unit is not None:
                self.__length_unit = get_length_unit(length_unit)
                self.__length_unit_name = get_length_unit_name(length_unit)
                self.__length_ref = get_length_ref(length_unit)
            if energy_unit is not None:
                self.__energy_unit = get_energy_unit(energy_unit)
                self.__energy_unit_name = get_energy_unit_name(energy_unit)
                self.__energy_ref = get_energy_ref(energy_unit)
        else:
            if not isinstance(units,Units):
                raise TypeError('The type of units must be "Units"')
            self.__length_unit = get_length_unit(units)
            self.__length_unit_name = get_length_unit_name(units)
            self.__length_ref = get_length_ref(units)
            self.__energy_unit = get_energy_unit(units)
            self.__energy_unit_name = get_energy_unit_name(units)
            self.__energy_ref = get_energy_ref(units)
        return self.set_depends()

    def set_depends(self):
        self.__boltzmann = _BOLTZMANN_DEFAULT_REF
        if self.__energy_ref is not None:
            self.__boltzmann /= self.__energy_ref
        self.__coulomb = _COULOMB_DEFAULT_REF
        if self.__length_ref is not None and self.__energy_ref is not None:
            self.__coulomb *= self.__energy_ref * self.__length_ref
        return self

    def length(self, value, unit=None):
        return value * self.convert_length_from(unit)

    def energy(self, value, unit=None):
        return value * self.convert_energy_from(unit)

    def convert_length_to(self, unit):
        return length_convert(self.__length_unit, unit)

    def convert_energy_to(self, unit):
        return energy_convert(self.__energy_unit, unit)

    def convert_length_from(self, unit):
        return length_convert(unit, self.__length_unit)

    def convert_energy_from(self, unit):
        return energy_convert(unit, self.__energy_unit)

    @property
    def boltzmann_def(self):
        return _BOLTZMANN_DEFAULT_REF

    @property
    def boltzmann(self):
        return self.__boltzmann

    @property
    def coulomb(self):
        return self.__coulomb

    @property
    def avogadro(self):
        return AVOGADRO_NUMBER

    @property
    def gas_constant(self):
        return GAS_CONSTANT

    @property
    def length_unit(self):
        return self.__length_unit

    @property
    def energy_unit(self):
        return self.__energy_unit

    @property
    def length_unit_name(self):
        return self.__length_unit_name

    @property
    def energy_unit(self):
        return self.__energy_unit

    @property
    def energy_unit_name(self):
        return self.__energy_unit_name

    @property
    def force_unit(self):
        return self.__energy_unit + '/' + self.__length_unit

    @property
    def velocity_unit(self):
        return self.__length_unit + '/ps'

    @property
    def length_ref(self):
        return self.__length_ref

    @property
    def energy_ref(self):
        return self.__energy_ref

    @property
    def force_ref(self):
        if self.__energy_ref is None:
            return None
        else:
            return self.__energy_ref / self.__length_ref

    @property
    def acceleration_ref(self):
        if self.__energy_ref is None or self.__length_ref is None:
            return None
        else:
            return self.__energy_ref / self.__length_ref / self.__length_ref

    @property
    def kinetic_ref(self):
        if self.__energy_ref is None or self.__length_ref is None:
            return None
        else:
            return self.__length_ref * self.__length_ref / self.__energy_ref

    @property
    def velocity_ref(self):
        if self.__energy_ref is None or self.__length_ref is None:
            return None
        else:
            return math.sqrt(self.__energy_ref / self.__length_ref / self.__length_ref)

global_units = Units('nm','kj/mol')

def set_global_length_unit(unit):
    global_units.set_length_unit(unit)

def set_global_energy_unit(unit):
    global_units.set_energy_unit(unit)

def set_global_units(length_unit,energy_unit,units=None):
    global_units.set_units(length_unit,energy_unit,units)