from __future__ import print_function
from __future__ import division
import os
import wikipedia
from argparse import ArgumentParser as AP

def _get_basic(pid, full_path):
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

p = AP()
p.add_argument('--root', type=str, default='./', help='Specify root location for storing data')
p.add_argument('--communities', type=str, required=True,
                                help='Specify location for community data. Expected structure:\
                                      <comm_id> \\t [person_id]+')
p.add_argument('--basic_info', type=str, default='./wsn_person-name-gender-birth-death.txt',
                                         help='File for obtaining the basic information')
p.parse_args()

basic_info = p.basic_info

new_dir = os.path.join(p.root, './raw_data')
if not os.path.exists(new_dir):
    os.mkdir(new_dir, mode=0o755)

c_id = 0
with open(p.communities, 'r') as all_comms:
    for community in all_comms:
        c_id += 1
        community_path = os.path.join(new_dir, 'community{}'.format(c_id))
        if not os.path.exists(community_path):
            os.mkdir(community_path)
        personids = community.split('\t')[1:]
        for pid in personids:
            details_dict = _get_basic(pid, basic_info)
            name = details_dict['name']
            if pid.find('WP') != -1:
                continue
            person_pagetext = os.path.join(community_path, '{}.txt'.format(pid))
            with open(person_pagetext, 'w') as pp:
                person_page = wikipedia.page(name)
                pp.write(person_page.content)
