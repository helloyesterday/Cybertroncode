import math
import numpy as np
import mindspore as ms

class Units:
    def __init__(self):

        self.float = ms.float32
        self.int = ms.int32

        # length
        self._length_def = 1.0
        self._length = self._length_def

        self._nm = self._length_def
        self._um = self._nm * 1e3
        self._angstrom = self._nm * 0.1
        self._bohr = self._nm * 0.052917721067

        self._length_dict = {
            'nm':self._nm,
            'um':self._um,
            'a':self._angstrom,
            'angstrom':self._angstrom,
            'bohr':self._bohr,
        }

        self._length_name = {
            'nm':'nm',
            'um':'um',
            'a':'Angstrom',
            'angstrom':'Angstrom',
            'bohr':'Bohr',
        }

        self._length_unit_def = 'nm'
        self._length_unit = self._length_unit_def

        # energy
        self._energy_def = 1.0
        self._energy = self._energy_def

        self._kj_mol = self._energy_def
        self._j_mol = self._kj_mol * 1e-3
        self._kcal_mol = self._kj_mol * 4.184
        self._cal_mol = self._kj_mol * 4.184e-3
        self._hartree = self._kj_mol * 2625.5002
        self._ev = self._kj_mol * 96.48530749925793

        self._energy_dict = {
            'kj/mol':self._kj_mol,
            'j/mol':self._j_mol,
            'kcal/mol':self._kcal_mol,
            'cal/mol':self._cal_mol,
            'ha':self._hartree,
            'hartree':self._hartree,
            'ev':self._ev,
        }

        self._energy_name = {
            'kj/mol':'kJ/mol',
            'j/mol':'J/mol',
            'kcal/mol':'Kcal/mol',
            'cal/mol':'cal/mol',
            'ha':'Hartree',
            'hartree':'Hartree',
            'ev':'eV',
        }

        self._energy_unit_def = 'kj/mol'
        self._energy_unit = self._energy_unit_def

        # origin constant
        self._avogadro_number = 6.02214076e23
        self._boltzmann_constant = 1.380649e-23
        self._gas_constant = 8.31446261815324
        self._elementary_charge = 1.602176634e-19
        self._coulomb_constant = 8.9875517923e9

        # Boltzmann constant
        self._boltzmann_def = 8.31446261815324e-3  # kj/mol
        self._boltzmann = self._boltzmann_def

        # Coulomb constant
        self._coulomb_def = 138.93545764498226165718756672623 # kj/mol*nm
        self._coulomb = self._coulomb_def

    def check_length_unit(self,unit):
        if unit.lower() not in self._length_dict.keys():
            raise ValueError('length unit "'+unit+'" is not recorded!')
        return self._length_name[unit.lower()]

    def check_energy_unit(self,unit):
        if unit.lower() not in self._energy_dict.keys():
            raise ValueError('energy unit "'+unit+'" is not recorded!')
        return self._energy_name[unit.lower()]

    def set_default(self):
        self._length_unit = self._length_unit_def
        self._length = self._length_def
        self._energy_unit = self._energy_unit_def
        self._energy = self._energy_def
        self._coulomb = self._coulomb_def
        self._boltzmann_def = self._boltzmann_def

    def set_length_unit(self,unit):
        self.check_length_unit(unit.lower())
        if unit.lower() != self._length_unit:
            self._length_unit = unit.lower()
            self._length = self._length_dict[unit.lower()]

            self._coulomb = self._coulomb_def \
                * self.def_energy_convert_to(self._energy_unit) \
                * self.def_length_convert_to(self._length_unit)

    def set_energy_unit(self,unit):
        self.check_energy_unit(unit.lower())
        if unit.lower() != self._energy_unit:
            self._energy_unit = unit.lower()
            self._energy = self._energy_dict[unit.lower()]
            
            self._boltzmann = self._boltzmann_def * self.def_energy_convert_to(unit.lower())
            self._coulomb = self._coulomb_def \
                * self.def_energy_convert_to(self._energy_unit) \
                * self.def_length_convert_to(self._length_unit)

    def length(self,_length,unit):
        self.check_length_unit(unit.lower())
        return _length * self._length_dict[unit.lower()] / self._length

    def energy(self,_energy,unit):
        self.check_energy_unit(unit.lower())
        return _energy * self._energy_dict[unit.lower()] / self._energy

    def length_convert(self,unit_in,unit_out):
        self.check_length_unit(unit_in.lower())
        self.check_length_unit(unit_out.lower())
        return self._length_dict[unit_in.lower()] / self._length_dict[unit_out.lower()]

    def energy_convert(self,unit_in,unit_out):
        self.check_energy_unit(unit_in.lower())
        self.check_energy_unit(unit_out.lower())
        return self._energy_dict[unit_in.lower()] / self._energy_dict[unit_out.lower()]

    def length_convert_to(self,unit):
        return self.length_convert(self._length_unit,unit.lower())

    def energy_convert_to(self,unit):
        return self.energy_convert(self._energy_unit,unit.lower())

    def def_length_convert_to(self,unit):
        return self.length_convert(self._length_unit_def,unit.lower())

    def def_energy_convert_to(self,unit):
        return self.energy_convert(self._energy_unit_def,unit.lower())

    def length_convert_from(self,unit):
        return self.length_convert(unit.lower(),self._length_unit)

    def energy_convert_from(self,unit):
        return self.energy_convert(unit.lower(),self._energy_unit)

    def Boltzmann_constant(self):
        return self._boltzmann_constant

    def Boltzmann(self,unit=None):
        if unit is None:
            return self._boltzmann
        else:
            return self._boltzmann_def * self.def_energy_convert_to(unit.lower())

    def Coulomb(self,energy_unit=None,length_unit=None):
        if (energy_unit is None) and (length_unit is None):
            return self._coulomb

        scale_energy = self.def_energy_convert_to(energy_unit.lower())
        scale_length = self.def_length_convert_to(length_unit.lower())
        return self._coulomb_def * scale_energy * scale_length

units = Units()
