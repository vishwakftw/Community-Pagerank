import os

# static values, could be changed before usage 
PATH = '../../CS6670/'
BASIC = 'wsn_person-name-gender-birth-death.txt'
OCCU = 'wsn_person-occupation.txt'

class Person(object):
    def __init__(self, person_id):
        self.person_id = person_id
        self.basic = None
        self.occupation = None

    def get_info(self, fields):
        if 'name' in fields:
            if self.basic is None:
                self.basic = _get_basic(self.person_id)
            print("Name = {}".format(self.basic['name']))

        if 'birth' in fields:
            if self.basic is None:
                self.basic = _get_basic(self.person_id)
            print("Birth Year = {}".format(self.basic['birth']))

        if 'death' in fields:
            if self.basic is None:
                self.basic = _get_basic(self.person_id)
            print("Death Year = {}".format(self.basic['death']))

        if 'gender' in fields:
            if self.basic is None:
                self.basic = _get_basic(self.person_id)
            print("Gender = {}".format(self.basic['gender']))

        if 'occupation' in fields:
            if self.occupation is None:
                self.occupation = _get_occupation(self.person_id)
            print("Occupation = {}".format(self.occupation))

    def reset(self):
        self.basic = None
        self.occupation = None

def _get_basic(pid, full_path=(PATH + BASIC)):
    details_dict = {}
    with open(full_path, 'r') as f:
        for line in f:
            vals = line.split('\t')
            if vals[0] == pid:
                break
    details_dict['name'] = vals[1]
    details_dict['birth'] = vals[3]
    details_dict['death'] = vals[4][:-1]  # newline character is present
    details_dict['gender'] = vals[2]
    return details_dict

def _get_occupation(pid, full_path=(PATH + OCCU)):
    with open(full_path, 'r') as f:
        for line in f:
            vals = line.split('\t')
            if vals[0] == pid:
                break
    return vals[1][:-1]  # newline character is present
