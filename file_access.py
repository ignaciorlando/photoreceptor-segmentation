
from os import makedirs

def natural_key(string_):
    '''
    To sort strings using natural ordering. See http://www.codinghorror.com/blog/archives/001018.html
    '''
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def parse_boolean(input_string):
    '''
    Parse a boolean
    '''
    return input_string.upper()=='TRUE'


def get_list_of_strings_from_string(given_string):
    '''
    Split a string with colons
    '''
    return given_string.replace(' ','').split(',')


def list_to_string(given_list):
    '''
    Get list as string
    '''
    return str(given_list).replace('[','').replace(']','').replace('\'','')


def makedir_if_doesnt_exist(full_path):
    '''
    Make the folder if it doesnt exist
    '''
    if not(path.exists(full_path)):
        makedirs(full_path)